import time, sys, os, json, argparse
from typing import List, Tuple, Dict
from tqdm import tqdm

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput

from req_wl import Workload, Request
from utils import read_info_from_csv, get_cur_info, print_requests, get_time, \
            init_time, final_time, cmp, get_metrics, print_metrics
from simulators import simulate_fcfs, simulate_interleave, simulate_sjmlfq, \
            simulate_sjmlfqmp, simulate_emlfq

path = '/users/ll/datasets/ShareGPT52K/sg_90k_part1.json'
max_length = 2048
requests_num = 50
magic_prompts_id = [11, 66]
sampling_params = SamplingParams(max_tokens=512)


def generate_workloads(
        args: argparse.Namespace, 
        prompts: List[str], 
        prompt_ids: List[int]
    ) -> Tuple[List[str], Dict[str, Workload]]: 
    # assume profiling data is stored in order in file info.csv
    assert prompt_ids == magic_prompts_id, "Only support this case now!"
    workload_types = len(prompt_ids)
    infos = read_info_from_csv("infos.csv")
    assert len(infos) == workload_types, \
        "length of info.csv does not match prompt_ids"
    test_prompts, workloads_dict = [], {}

    for num in range(workload_types):
        workload_type = "job" + str(num)
        # add prompt to workload info
        infos[num]["prompt"] = prompts[num]
        # add profiling data to workload info
        get_cur_info(args.tensor_parallel_size, infos[num])
        # create workload
        workload = Workload(workload_type, infos[num])
        # add the type of workload to workloads_dict
        workloads_dict[workload_type] = workload
        test_prompts.append(prompts[num])
    return test_prompts, workloads_dict

def generate_requests(workloads_dict: Dict[str, Workload]) -> List[Request]:
    if args.workload_type == "maf1":
        from alpa_serve.trace import Trace

        train_start = "0.0.0"
        train_end = "0.1.0"
        azure_v1_trace_dir = "trace/azure_v1.pkl"
        azure_v1_trace = Trace("azure_v1", azure_v1_trace_dir)
        rate_scale = args.rate_scale
        num_models = 2
        model_names = [f"job{i}" for i in range(num_models)]
        train_replays = azure_v1_trace.replay(model_names, model_mapping_strategy="round_robin", 
            arrival_distribution="gamma", start_time=train_start, end_time=train_end, 
            interval_seconds=60, rate_scale_factor=rate_scale, cv_scale_factor=1)

        # for x in train_replays.arrivals:
        arrival_time = []
        for model_name in model_names:
            for arrival in train_replays[model_name].arrivals:
                arrival_time.append((arrival, model_name))
        arrival_time.sort(key=lambda x:x[0])

        requests = []
        cur_rid = 0
        for x in arrival_time:
            stamp, w_type = x
            w_info = workloads_dict[w_type].info_args
            r_time = w_info["t_in"]+w_info["t_out"]*w_info["st_len_out"]
            # todo: consider slo for testbed
            slo = r_time*args.slo_rate if not args.simulate else float("inf")
            request = Request(cur_rid, stamp, w_type, 0, w_info["st_len_out"], slo)
            cur_rid += 1
            requests.append(request)

        return requests

# return questions from human in sharegpt dataset. 
def create_test_prompts() -> List[str]:
    prompts = []
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for entry in data: 
        for topic in entry['conversations']:
            if topic['from'] == 'human':
                if len(topic['value']) < max_length and len(topic['value']) != 0:
                    prompts.append(topic['value'])

    return prompts


# add one time and step one time.
def warmup_process_requests(engine: LLMEngine,
                     test_prompts: List[str]) -> None:
    """Continuously process a list of prompts and handle the outputs."""

    request_id = 0
    iter_count = 1
    time_list = []
    input_len = [0]*len(test_prompts)
    output_len = [0]*len(test_prompts)
    # output2_len = [0]*len(test_prompts)
    while test_prompts or engine.has_unfinished_requests():
        time0 = time.perf_counter()
        if test_prompts:
            prompt = test_prompts.pop(0)
            # input_len.append(len(prompt))
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()
        
        for request_output in request_outputs:
            if request_output.finished:
                # output_len.append(len(request_output.prompt))
                time_list.append(request_output.metrics.finished_time-request_output.metrics.arrival_time)
                input_len[int(request_output.request_id)] = len(request_output.prompt_token_ids)
                output_len[int(request_output.request_id)] = len(request_output.outputs[0].token_ids)
                # exit()
        time1 = time.perf_counter()
        # print(time1-time0)

        iter_count += 1

    print("\nnum of requests processed:", len(time_list))
    print("avg time:", sum(time_list)/len(time_list))
    print("block_size = ", engine.cache_config.block_size)
    for aa, bb in zip(input_len, output_len):
        print(aa, bb)
    import csv
    with open("test.csv", "w") as file:
        writer = csv.writer(file)
        for aa, bb in zip(input_len, output_len):
            writer.writerow([aa,bb])

    return sum(time_list)/len(time_list)

def get_new_requests(requests, next_rid, cur_time):
    """Return newly-arrival requests"""
    return_req = []
    while next_rid<len(requests):
        if cmp(requests[next_rid].arrival_time, cur_time)<=0:
            return_req.append(requests[next_rid])
        else:
            break
        next_rid += 1
    return return_req, next_rid

# add according to arrival time and step one time.
def process_requests(
        engine: LLMEngine,
        requests: List[Request],
        workloads_dict: Dict[str, Workload]
    ) -> None:
    """Continuously process a list of prompts and handle the outputs."""

    request_id = 0
    iter_count = 1
    time_list = []
    requests.sort(key=lambda x: x.arrival_time)
    next_rid = 0
    zero_time = init_time()
    # global zero_time
    global one_time
    # print(next_rid, len(requests))
    while next_rid<len(requests) or engine.has_unfinished_requests():
        # time0 = time.perf_counter()
        cur_time = get_time(zero_time)
        new_reqs, next_rid = get_new_requests(requests, next_rid, cur_time)
        for req in new_reqs:
            workload_info = workloads_dict[req.workload_type].info_args
            prompt = workload_info["prompt"]
            engine.add_request(str(request_id), prompt, sampling_params, workload_info=workload_info) # todo  slo
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()
        # print("process_request:", [request_output.request_id])
        for request_output in request_outputs:
            # print(len(request_output.prompt_token_ids))
            if request_output.finished:
                # if int(request_output.request_id) in [2, 5, 7]:
                #     print("123123", request_output.request_id, request_output.prompt_token_ids, request_output.outputs[0].token_ids)
                time_list.append(request_output.metrics.finished_time-request_output.metrics.arrival_time)
                finished_rid = int(request_output.request_id)
                requests[finished_rid].finish_time = request_output.metrics.finished_time-zero_time
                requests[finished_rid].latency = requests[finished_rid].finish_time-requests[finished_rid].arrival_time
                requests[finished_rid].input_len = len(request_output.prompt_token_ids)
                requests[finished_rid].output_len = len(request_output.outputs[0].token_ids)
        time1 = time.perf_counter()
        # print(time1-time0, next_rid)

        iter_count += 1
    one_time = final_time()

    metrics = get_metrics(args, requests, workloads_dict, one_time-zero_time)
    print_metrics(args, workloads_dict, metrics)
    # for val in output_len:
    #     print(val)

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    # parser.add_argument("--policy", choices=["fcfs", "interleave", "sjmlfq", "emlfq", "sjmlfqmp"])
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--workload-type", choices=["maf1", "maf2"])
    parser.add_argument("--rate-scale", type=float, default=0.1)
    parser.add_argument("--slo-rate", type=float, default="5")
    parser.add_argument("--output-filename", type=str, default="result")
    parser.add_argument("--strict-stop", action='store_true')
    args = parser.parse_args()
    # todo: consider slo for testbed exp
   
    one_time = -1

    print("Start generate requests!")
    all_prompts = create_test_prompts()
    test_prompts, workloads_dict = generate_workloads(args, all_prompts, magic_prompts_id)
    requests = generate_requests(workloads_dict)
    real_requests = requests[:requests_num]
    print("End generating requests\nStart process requests!")

    if args.simulate: # simulator    
        if args.scheduler_policy == "fcfs":
            metrics, max_kv = simulate_fcfs(requests, workloads_dict)
        elif args.scheduler_policy == "interleave":
            metrics, max_kv = simulate_interleave(requests, workloads_dict)
        elif args.scheduler_policy == "sjmlfq":
            metrics, max_kv = simulate_sjmlfq(requests, workloads_dict)
        elif args.scheduler_policy == "sjmlfqmp":
            metrics, max_kv = simulate_sjmlfqmp(requests, workloads_dict)
        elif args.scheduler_policy == "emlfq":
            metrics, max_kv = simulate_emlfq(requests, workloads_dict)
    else:
        # testbed
        # zero_time = init_time()
        engine = initialize_engine(args)
        process_requests(engine, real_requests, workloads_dict)
        # warmup_process_requests(engine, [test_prompts[1], test_prompts[1]])
        print_requests(real_requests)
