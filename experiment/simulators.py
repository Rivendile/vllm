from utils import get_metrics, EPS

def simulate_fcfs(requests, w_dict):
    tmp_device_time = 0
    max_kv = 0
    for r in requests:
        tmp_device_time = max(r.arrival_time, tmp_device_time)
        while True:
            next_time = r.get_next_process_time()
            if tmp_device_time+next_time-r.arrival_time<=r.slo+EPS:
                tmp_device_time += next_time
                finish_flag = r.process_token()
                max_kv = max(max_kv, r.cur_token)
                if finish_flag:
                    r.finish_time = tmp_device_time
                    r.latency = r.finish_time - r.arrival_time
                    break
            else:
                break

    return get_metrics(requests), max_kv

def calc_kv(requests_queue):
    sum_kv = 0
    for r in requests_queue:
        sum_kv += max(0, r.cur_token)
    return sum_kv

def simulate_interleave(requests, w_dict):
    tmp_device_time = 0
    next_event_time = requests[0].arrival_time
    requests_queue = []
    cur_rid = 0
    max_kv = 0
    while True:
        if cur_rid>=len(requests) and len(requests_queue)==0:
            # finish all requests
            break

        # process past event
        if len(requests_queue)>0:
            if tmp_device_time+requests_queue[0].get_next_process_time()<=next_event_time:
                finish_flag = requests_queue[0].process_token()
                tmp_time = tmp_device_time+requests_queue[0].get_next_process_time()
                if finish_flag:
                    if tmp_time-requests_queue[0].arrival_time<=requests_queue[0].slo:
                        requests_queue[0].finish_time = tmp_time
                        requests_queue[0].latency = tmp_time - requests_queue[0].arrival_time
                    requests_queue.pop(0)
                else:
                    r = requests_queue[0]
                    requests_queue.pop(0)
                    if tmp_time-r.arrival_time<=r.slo:
                        requests_queue.append(r)
            max_kv = max(max_kv, calc_kv(requests_queue))
        
        tmp_device_time = next_event_time
        # add new event
        while cur_rid<len(requests) and requests[cur_rid].arrival_time<=tmp_device_time:
            requests_queue.append(requests[cur_rid])
            cur_rid+=1
        
        # get next event
        next_event_time = float("inf")
        if cur_rid<len(requests):
            next_event_time = min(next_event_time, requests[cur_rid].arrival_time)
        if len(requests_queue)>0:
            next_event_time = min(next_event_time, tmp_device_time+requests_queue[0].get_next_process_time())
    
    return get_metrics(requests), max_kv
        
def get_priority(last_prio, cur_itertime):
    cur_itertime = int(cur_itertime / TIME_UNIT)
    new_prio = 1 if last_prio==0 else last_prio*2
    while cur_itertime>new_prio:
        new_prio *= 2
    return new_prio

def simulate_sjmlfq(requests_all, w_dict):
    max_kv = 0
    for t_job in workloads_dict.keys():
        requests = []
        for req in requests_all:
            if req.workload_type == t_job:
                requests.append(req)
        if len(requests)==0:
            continue

        tmp_device_time = requests[0].arrival_time
        requests_queue = {}
        cur_rid = 0
        starvation_limit = args.workload_duration*SLO_RATE
        while True:
            left_requests = 0
            for prio, val in requests_queue.items():
                left_requests += len(val)
            if cur_rid>=len(requests) and left_requests == 0:
                # finish all requests
                break
            
            # print(cur_rid, len(requests), len(requests_queue), tmp_device_time)

            # add new requests
            while cur_rid<len(requests) and requests[cur_rid].arrival_time<=tmp_device_time:
                cur_prio = get_priority(0, requests[cur_rid].input_len*workloads_dict[requests[cur_rid].workload_type].info_args["t_in"])
                requests[cur_rid].priority = cur_prio
                requests[cur_rid].prio_quan = cur_prio
                if cur_prio not in requests_queue:
                    requests_queue[cur_prio] = []
                requests_queue[cur_prio].append(requests[cur_rid])
                cur_rid += 1
            
            # process requests
            for prio, rs in requests_queue.items():
                pr_rs = []
                for r in rs:
                    if tmp_device_time - r.arrival_time <= r.slo:
                        pr_rs.append(r)
                requests_queue[prio] = rs

            # demote requests
            add_prio = []
            for prio, rs in requests_queue.items():
                for r in rs:
                    next_iter_time = r.get_next_process_time(tp=False)
                    if r.prio_quan<next_iter_time/TIME_UNIT:
                        old_prio = r.priority
                        new_prio = get_priority(old_prio, next_iter_time)
                        if new_prio not in requests_queue:
                            add_prio.append(new_prio)
            for prio in add_prio:
                requests_queue[prio] = []
            for prio, rs in requests_queue.items():
                pr_rs = []
                for r in rs:
                    next_iter_time = r.get_next_process_time(tp=False)
                    if r.prio_quan<next_iter_time/TIME_UNIT:
                        old_prio = r.priority
                        r.priority = get_priority(old_prio, next_iter_time)
                        r.prio_quan = r.priority
                        assert r.priority in requests_queue
                        requests_queue[r.priority].append(r)
                    else:
                        pr_rs.append(r)
                requests_queue[prio] = pr_rs
            # promote requests
            for prio, rs in requests_queue.items():
                pr_rs = []
                if prio==1:
                    continue
                for r in rs:
                    if tmp_device_time - r.last_exec_time >= starvation_limit:
                        r.priority = 1
                        r.prio_quan = r.get_next_process_time(tp=False)
                        requests_queue[r.priority].append(r)
                    else:
                        pr_rs.append(r)
                requests_queue[prio] = pr_rs
            
            # calc kv
            tmp_kv = 0
            for prio, rs in requests_queue.items():
                for r in rs:
                    tmp_kv = max(tmp_kv, r.cur_token)
            max_kv = max(max_kv, tmp_kv)
            
            # execute requests
            process_flag = False
            for prio, rs in requests_queue.items():
                if len(rs)==0:
                    continue
                r = rs[0]
                next_iter_time = r.get_next_process_time(tp=False)
                tmp_device_time += next_iter_time
                r.last_exec_time = tmp_device_time
                finish_flag = r.process_token()
                r.prio_quan -= next_iter_time / TIME_UNIT
                assert r.prio_quan >= 0
                if tmp_device_time - r.arrival_time > r.slo:
                    rs.pop(0)
                elif finish_flag:
                    r.finish_time = tmp_device_time
                    r.latency = r.finish_time - r.arrival_time
                    rs.pop(0)
                process_flag = True
                break
            if not process_flag:
                if cur_rid < len(requests):
                    tmp_device_time = requests[cur_rid].arrival_time


    return get_metrics(requests_all), max_kv

def simulate_sjmlfqmp(requests, w_dict):
    tmp_device_time = requests[0].arrival_time
    requests_queue = {}
    cur_rid = 0
    max_kv = 0
    starvation_limit = args.workload_duration*SLO_RATE
    while True:
        left_requests = 0
        for prio, val in requests_queue.items():
            left_requests += len(val)
        if cur_rid>=len(requests) and left_requests == 0:
            # finish all requests
            break
        
        # print(cur_rid, len(requests), len(requests_queue), tmp_device_time)

        # add new requests
        while cur_rid<len(requests) and requests[cur_rid].arrival_time<=tmp_device_time:
            cur_prio = get_priority(0, requests[cur_rid].input_len*workloads_dict[requests[cur_rid].workload_type].info_args["tp_t_in"])
            requests[cur_rid].priority = cur_prio
            requests[cur_rid].prio_quan = cur_prio
            if cur_prio not in requests_queue:
                requests_queue[cur_prio] = []
            requests_queue[cur_prio].append(requests[cur_rid])
            cur_rid += 1
        
        # process requests
        for prio, rs in requests_queue.items():
            pr_rs = []
            for r in rs:
                if tmp_device_time - r.arrival_time <= r.slo:
                    pr_rs.append(r)
            requests_queue[prio] = rs

        # demote requests
        add_prio = []
        for prio, rs in requests_queue.items():
            for r in rs:
                next_iter_time = r.get_next_process_time()
                if r.prio_quan<next_iter_time/TIME_UNIT:
                    old_prio = r.priority
                    new_prio = get_priority(old_prio, next_iter_time)
                    if new_prio not in requests_queue:
                        add_prio.append(new_prio)
        for prio in add_prio:
            requests_queue[prio] = []
        for prio, rs in requests_queue.items():
            pr_rs = []
            for r in rs:
                next_iter_time = r.get_next_process_time()
                if r.prio_quan<next_iter_time/TIME_UNIT:
                    old_prio = r.priority
                    r.priority = get_priority(old_prio, next_iter_time)
                    r.prio_quan = r.priority
                    assert r.priority in requests_queue
                    requests_queue[r.priority].append(r)
                else:
                    pr_rs.append(r)
            requests_queue[prio] = pr_rs
        # promote requests
        for prio, rs in requests_queue.items():
            pr_rs = []
            if prio==1:
                continue
            for r in rs:
                if tmp_device_time - r.last_exec_time >= starvation_limit:
                    r.priority = 1
                    r.prio_quan = r.get_next_process_time()
                    requests_queue[r.priority].append(r)
                else:
                    pr_rs.append(r)
            requests_queue[prio] = pr_rs
        
        # calc kv
        tmp_kv = 0
        for prio, rs in requests_queue.items():
            for r in rs:
                tmp_kv = max(tmp_kv, r.cur_token)
        max_kv = max(max_kv, tmp_kv)
        
        # execute requests
        process_flag = False
        for prio, rs in requests_queue.items():
            if len(rs)==0:
                continue
            r = rs[0]
            next_iter_time = r.get_next_process_time()
            tmp_device_time += next_iter_time
            r.last_exec_time = tmp_device_time
            finish_flag = r.process_token()
            r.prio_quan -= next_iter_time / TIME_UNIT
            assert r.prio_quan >= 0
            if tmp_device_time - r.arrival_time > r.slo:
                rs.pop(0)
            elif finish_flag:
                r.finish_time = tmp_device_time
                r.latency = r.finish_time - r.arrival_time
                rs.pop(0)
            process_flag = True
            break
        if not process_flag:
            if cur_rid < len(requests):
                tmp_device_time = requests[cur_rid].arrival_time


    return get_metrics(requests), max_kv
    

def simulate_emlfq(requests, w_dict):
    tmp_device_time = requests[0].arrival_time
    requests_queue = []
    cur_rid = 0
    max_kv = 0
    starvation_limit = args.workload_duration*SLO_RATE
    # To do
    while True:
        if cur_rid>=len(requests) and len(requests_queue) == 0:
            # finish all requests
            break
        
        # print(tmp_device_time, cur_rid, len(requests), len(requests_queue))
        # add new requests
        while cur_rid<len(requests) and requests[cur_rid].arrival_time<=tmp_device_time+EPS:
            w_type = requests[cur_rid].workload_type
            requests[cur_rid].priority = requests[cur_rid].input_len*workloads_dict[w_type].info_args["tp_t_in"]+workloads_dict[w_type].info_args["st_len_out"]*workloads_dict[w_type].info_args["tp_t_out"] *(1+2*args.output_var)
            requests_queue.append(requests[cur_rid])
            requests[cur_rid].prio_quan = requests[cur_rid].priority
            cur_rid += 1
        
        # print(tmp_device_time, cur_rid, len(requests), len(requests_queue))

        pr_requests_queue = []
        for r in requests_queue:
            if tmp_device_time - r.arrival_time <= r.slo+EPS:
                pr_requests_queue.append(r)
        requests_queue = pr_requests_queue

        # calc kv
        tmp_kv = 0
        for r in requests_queue:
            tmp_kv = max(tmp_kv, r.cur_token)
        max_kv = max(max_kv, tmp_kv)

        # execute requests
        process_flag = False
        while len(requests_queue)>0 and not process_flag:
            requests_queue.sort(key=lambda x: x.priority)
            r = requests_queue[0]
            next_iter_time = r.get_next_process_time()
            if r.priority - next_iter_time >= -EPS:
                process_flag = True
                tmp_device_time += next_iter_time
                r.priority -= next_iter_time
                finish_flag = r.process_token()
                # print("process: ", r.rid, r.priority, next_iter_time, finish_flag)
                if finish_flag:
                    r.finish_time = tmp_device_time
                    r.latency = r.finish_time - r.arrival_time
                    requests_queue.pop(0)
            else:
                r.priority = r.prio_quan * 2
                r.prio_quan = r.priority
        
        if not process_flag:
            if cur_rid < len(requests):
                tmp_device_time = requests[cur_rid].arrival_time
        
    return get_metrics(requests), max_kv
