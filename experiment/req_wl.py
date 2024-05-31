class Workload:
    def __init__(self, workload_type, info_args):
        self.workload_type = workload_type
        self.info_args = info_args


class Request:
    def __init__(self, rid, arrival_time, workload_type, input_len, output_len, slo):
        self.rid = rid
        self.arrival_time = arrival_time
        self.workload_type = workload_type
        self.input_len = input_len
        self.output_len = output_len
        self.finish_time = None
        self.latency = None
        self.slo = slo
        self.cur_token = -1 # -1 for not served; 0 for finish input; 1-n for output token
        self.priority = None
        self.prio_quan = None
        self.last_exec_time = arrival_time
