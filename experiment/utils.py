import time
import csv
from typing import List, Tuple, Dict
from req_wl import Workload, Request

def print_requests(requests):
    print(f"{len(requests)} in all:")
    for req in requests:
        print(req.rid, req.workload_type, req.input_len, req.output_len, req.arrival_time, req.slo, req.finish_time, req.latency)

def init_time():
    return time.time()

def final_time():
    return time.time()

def get_time(zero_time):
    assert zero_time>=0, "Not initiate time!"
    return time.time()-zero_time

EPS = 1e-6
def cmp(x,y):
    if x-y>EPS:
        return 1
    elif y-x>EPS:
        return -1
    else:
        return 0
    
def get_metrics(
        args, 
        requests: List[Request], 
        workloads_dict: Dict[str, Workload], 
        workload_duration: float
    ) -> Dict[str, Dict[str, float]]:
    latencies = {}  # latencies of different workload_types
    tokens = {}     # output length of different workload_types
    duration = {}   # duration of different workload_types
    norm_latencies = {} 
    counts = {}     # Counts of different work types, including requests that are not finished
    for key in workloads_dict.keys():
        duration[key] = workload_duration
    for r in requests:
        if r.finish_time != None :
            if r.workload_type not in latencies:
                latencies[r.workload_type] = []
                tokens[r.workload_type] = []
                duration[r.workload_type] = 0
                norm_latencies[r.workload_type] = []
            latencies[r.workload_type].append(r.latency)
            w_info = workloads_dict[r.workload_type].info_args
            rtime = w_info["t_in"]+w_info["t_out"]*w_info["st_len_out"]
            norm_latencies[r.workload_type].append(r.latency/rtime)
            tokens[r.workload_type].append(r.output_len)
            duration[r.workload_type] = max(duration[r.workload_type], r.finish_time)
        if r.workload_type not in counts:
            counts[r.workload_type] = 0
        counts[r.workload_type] += 1
    
    metrics = {}
    all_latency = []
    all_norm_latency = []
    all_duration = workload_duration
    all_tokens = []
    all_counts = len(requests)
    for key, val in latencies.items():
        val.sort()
        metrics[key]={}
        metrics[key]["avg_latency"] = sum(val)/len(val)
        metrics[key]['norm_latency'] = sum(norm_latencies[key])/len(norm_latencies[key])
        metrics[key]["p99_latency"] = val[int(len(val)*0.99)]
        metrics[key]["request_tput"] = len(val)/duration[key] if duration[key]>0 else 0
        metrics[key]["token_tput"] = sum(tokens[key])/duration[key] if duration[key]>0 else 0
        metrics[key]["slo_attainment"] = len(val)/counts[key]

        all_latency.extend(val)
        all_norm_latency.extend(norm_latencies[key])
        all_duration = max(all_duration, duration[key])
        all_tokens.extend(tokens[key])
        # all_counts += counts[key]
    
    metrics["overall"] = {}
    metrics["overall"]["avg_latency"] = sum(all_latency)/len(all_latency) if len(all_latency)>0 else 0
    metrics["overall"]["norm_latency"] = sum(all_norm_latency)/len(all_norm_latency) if len(all_norm_latency)>0 else 0
    metrics["overall"]["p99_latency"] = all_latency[int(len(val)*0.99)] if len(all_latency)>0 else 0
    metrics["overall"]["request_tput"] = len(all_latency)/all_duration
    metrics["overall"]["token_tput"] = sum(all_tokens)/all_duration
    metrics["overall"]["slo_attainment"] = len(all_latency)/all_counts if all_counts>0 else 0

    # print(latencies)
    # print(all_latency)
    # print(all_norm_latency)
    # print(norm_latencies)
    
    return metrics

    
def print_metrics(args, workloads_dict, metrics):
    # print(metrics)
    # print("\n\n\n\n")
    print("----------\n", args.scheduler_policy)
    for key in list(workloads_dict.keys())+["overall"]:
        if key in metrics:
            val = metrics[key]
            if key!="overall":
                print(key, workloads_dict[key].info_args)
            print(f"{key}: avg latency: {val['avg_latency']}, p99 latency: {val['p99_latency']}, request tput: {val['request_tput']}, token tput: {val['token_tput']}, slo attainment: {val['slo_attainment']}")
        else:
            print(f"{key}: no such requests")
            print(key, workloads_dict[key].info_args)
            print(f"{key}: avg latency: 0, p99 latency: 0, request tput: 0, token tput: 0, slo attainment: 0")

    # print(f"Max kv used: {max_kv}")

    with open(args.output_filename+".csv", "a") as csvfile:
        writer = csv.writer(csvfile)

        write_content = [args.scheduler_policy, args.rate_scale]
        for metric_str in ["avg_latency", "norm_latency", "p99_latency", "request_tput", "token_tput", "slo_attainment"]:
            for val_str in ["job0", "job1", "overall"]:
                if val_str in metrics:
                    val = metrics[val_str][metric_str]
                else:
                    val = 0
                write_content.append(val)
        writer.writerow(write_content)
