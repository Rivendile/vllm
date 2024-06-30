from typing import Dict

"""
parameters in info_args in class Workload:
"tp_t_in"       : prefill time when tp = 2
"tp_t_out"      : time to decode a token when tp = 2  
"st_len_out"    : length of output token
"t_in"          : prefill time when tp = 1
"t_out"         : time to decode a token when tp = 1
"prompt_id"     : id of prompt
"cur_t_in"      : current prefill time (depending on args)
"cur_t_out"     : current time to decode a token (depending on args)
"""


class Workload:
    def __init__(
            self, 
            workload_type: str, 
            info_args: Dict[str, float]
        ) -> None:
        self.workload_type = workload_type
        self.info_args = info_args

    def __repr__(self) -> str:
        args_str = ', '.join(f"{key}={value}" for key, value in self.info_args.items() if key != "prompt")
        return f"Workload(workload_type={self.workload_type}, info_args={args_str})"
    
    def __str__(self) -> str:
        return repr(self)


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

    def __repr__(self) -> str:
        return (f"Request(rid={self.rid}, arrival_time={self.arrival_time}, workload_type={self.workload_type}, "
        f"input_len={self.input_len}, output_len={self.output_len}, finish_time={self.finish_time}, "
        f"latency={self.latency}, slo={self.slo}, cur_token={self.cur_token}, "
        f"priority={self.priority}, prio_quan={self.prio_quan}, last_exec_time={self.last_exec_time})")
    
    def __str__(self) -> str:
        return repr(self)

