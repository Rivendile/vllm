import time, sys, os, json, argparse
from typing import List, Tuple
from tqdm import tqdm

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm import LLM

from req_wl import Workload, Request
from utils import print_requests, get_time, init_time, final_time, cmp, get_metrics, print_metrics

path = '/users/zyh/datasets/ShareGPT52K/sg_90k_part1.json'
max_length = 2048
max_test_num = 500
sampling_params = SamplingParams(max_tokens=256)


def generate_workloads(prompts, prompt_ids):
    # print(len(prompts[0]), len(prompts[21]))
    if prompt_ids == [0,21]:
        #todo: tp need to profile
        info0 = {"tp_t_in": 0.0495, "tp_t_out": 0.0136, "st_len_out": 539, "t_in": 0.0495, "t_out":0.0136, "prompt": prompts[0]}
        info1 = {"tp_t_in": 0.0188, "tp_t_out": 0.0129, "st_len_out": 22, "t_in": 0.0188, "t_out":0.0129, "prompt": prompts[21]}
        workload0 = Workload("job0", info0)
        workload1 = Workload("job1", info1)
        workloads_dict = {"job0":workload0, "job1":workload1}

        test_prompts = []
        for prompt_id in prompt_ids:
            test_prompts.append(prompts[prompt_id])
        return test_prompts, workloads_dict
    else:
        raise NotImplementedError

def generate_requests(workloads_dict):
    if args.workload_type == "maf1":
        from alpa_serve.trace import Trace

        train_start = "0.0.0"
        train_end = "0.1.0"
        azure_v1_trace_dir = "trace/azure_v1.pkl"
        azure_v1_trace = Trace("azure_v1", azure_v1_trace_dir)
        rate_scale = args.rate_scale
        num_models = 2
        model_names = [f"job{i}" for i in range(num_models)]
        train_replays = azure_v1_trace.replay(model_names, model_mapping_strategy="round_robin", arrival_distribution="gamma", start_time=train_start, end_time=train_end, interval_seconds=60, rate_scale_factor=rate_scale, cv_scale_factor=1)

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
            slo = r_time*args.slo_rate
            request = Request(cur_rid, stamp, w_type, w_info["st_len_out"], slo)
            cur_rid += 1
            requests.append(request)

        return requests


def create_test_prompts() -> List[str]:
    prompts = []
    count = 0
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for entry in data: 
        for topic in entry['conversations']:
            if topic['from'] == 'human':
                if len(topic['value']) < max_length and len(topic['value']) != 0:
                    prompts.append(topic['value'])
                    count += 1
                    if count >= max_test_num:
                        return prompts


# add one time and step one time.
def warmup_process_requests(engine: LLMEngine,
                     test_prompts: List[str]):
    """Continuously process a list of prompts and handle the outputs."""

    request_id = 0
    iter_count = 1
    time_list = []
    input_len = []
    output_len = [0]*len(test_prompts)
    output2_len = [0]*len(test_prompts)
    while test_prompts or engine.has_unfinished_requests():
        # time0 = time.perf_counter()
        if test_prompts:
            prompt = test_prompts.pop(0)
            input_len.append(len(prompt))
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()
        
        for request_output in request_outputs:
            if request_output.finished:
                # output_len.append(len(request_output.prompt))
                time_list.append(request_output.metrics.finished_time-request_output.metrics.arrival_time)
                output_len[int(request_output.request_id)] = len(request_output.prompt_token_ids)
                output2_len[int(request_output.request_id)] = len(request_output.prompt)
        # time1 = time.perf_counter()
        # print(time1-time0)

        iter_count += 1

    print("\nnum of requests processed:", len(time_list))
    print("avg time:", sum(time_list)/len(time_list))
    print("block_size = ", engine.cache_config.block_size)
    # for aa, bb, cc in zip(input_len, output_len, output2_len):
    #     print(aa, bb, cc)
    import csv
    with open("test.csv", "w") as file:
        writer = csv.writer(file)
        for aa, bb, cc in zip(input_len, output_len):
            writer.writerow([aa,bb,cc])

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
def process_requests(engine: LLMEngine,
                    requests: List[Request],
                    workloads_dict):
    """Continuously process a list of prompts and handle the outputs."""

    request_id = 0
    iter_count = 1
    time_list = []
    requests.sort(key=lambda x: x.arrival_time)
    next_rid = 0
    global zero_time
    global one_time
    # input_len = []
    output_len = []
    while next_rid<len(requests) or engine.has_unfinished_requests():
        # time0 = time.perf_counter()
        cur_time = get_time(zero_time)
        new_reqs, next_rid = get_new_requests(requests, next_rid, cur_time)
        for req in new_reqs:
            workload_info = workloads_dict[req.workload_type].info_args
            prompt = workload_info["prompt"]
            engine.add_request(str(request_id), prompt, sampling_params) # todo  slo
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()
        
        for request_output in request_outputs:
            if request_output.finished:
                time_list.append(request_output.metrics.finished_time-request_output.metrics.arrival_time)
                finished_rid = int(request_output.request_id)
                output_len.append([len(request_output.prompt), requests[finished_rid].workload_type])
                requests[finished_rid].finish_time = request_output.metrics.finished_time-zero_time
                requests[finished_rid].latency = requests[finished_rid].finish_time-requests[finished_rid].arrival_time
                print(request_output.request_id, len(request_output.prompt_token_ids))
        # time1 = time.perf_counter()
        # print(time1-time0)

        iter_count += 1
    one_time = final_time()

    metrics = get_metrics(args, requests, workloads_dict, one_time-zero_time)
    print_metrics(args, workloads_dict, metrics)
    # for val in output_len:
    #     print(val)
    return sum(time_list)/len(time_list)

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument("--policy", choices=["fcfs", "interleave", "sjmlfq", "emlfq"])
    parser.add_argument("--workload-type", choices=["maf1", "maf2"])
    parser.add_argument("--rate-scale", type=float, default=0.1)
    parser.add_argument("--slo-rate", type=float, default="5")
    parser.add_argument("--output-filename", type=str, default="test_gpus")
    args = parser.parse_args()
    engine = initialize_engine(args)

    print("warming up...")
    tmp_prompts = create_test_prompts()
    warmup_process_requests(engine, tmp_prompts)
    print("warming up done.")
    one_time = -1
    # exit()

    print("Start generate requests!")
    tmp_prompts = create_test_prompts()
    test_prompts, workloads_dict = generate_workloads(tmp_prompts, [0,21])
    requests = generate_requests(workloads_dict)
    print_requests(requests[:10])
    print("Start process requests!")

    zero_time = init_time()
    process_requests(engine, requests[:10], workloads_dict)

    # todo: output len from len(prompt) to len(tokens)