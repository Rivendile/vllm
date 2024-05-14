"""Selective replication with model parallelism."""
from collections import namedtuple
from functools import partial
import logging
import math
import multiprocessing
import time
from typing import List, Tuple

import numpy as np
import ray

from alpa_serve.profiling import ParallelConfig
from alpa_serve.placement_policy.base_policy import (
    BasePlacementPolicy, ModelData, ClusterEnv, ModelPlacement, 
    ModelPlacementWithReplacement,
    PlacementEvaluator, gen_train_workload,
    replica_placement_round_robin, replica_placement_fast_greedy_llm,
    replica_placement_fast_greedy, replica_placement_beam_search,
    replica_placement_on_last_group, evolutionary_search)
from alpa_serve.simulator.controller import simulate_one_case
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload, GammaProcess
from alpa_serve.trace import Trace
from alpa_serve.util import (
    get_factors, get_partitions, get2tok, decompose2tok,
    ServingCase, eps)


def compute_capability(model_data, parallel_config, max_bs):
    slo = model_data.slo
    latency_mem = model_data.profiling_result.para_dict.get(parallel_config, None)

    if latency_mem is None:
        return 0

    num_stages = parallel_config.pp
    max_cap = 0
    for b, ls in latency_mem.latency.items():
        if b > max_bs:
            continue

        # slo = sum(ls) + (n-1) * max(ls)
        # so, n = ceil((slo - sum(ls)) / max(ls)) + 1
        max_cap = max(max_cap, (slo - sum(ls)) // max(ls) + 1)

    return max_cap * (0.99 ** num_stages)


class AlpaserveLLMGreedy(BasePlacementPolicy):
    
    def __init__(self,
                 max_bs: int = 1,
                 max_pp: int = 8,
                 max_op: int = 4,
                 use_evo_search: bool = False,
                 use_separation: bool = False,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_bs = max_bs
        self.max_pp = max_pp
        self.max_op = max_op
        self.n_iter = 1
        self.seed = 0
        self.beam_size = 3
        self.use_evo_search = use_evo_search
        self.use_separation = use_separation

        self.evaluator_method = "fast_simulator"
        self.parallel_evaluator = False
        self.parallel_initial_placement = False

        if ((self.parallel_evaluator or self.parallel_initial_placement)
            and not ray.is_initialized()):
            ray.init(address="auto", ignore_reinit_error=True)


    def solve_placement_one_eco(self,
                                model_datas: List[ModelData],
                                cluster_env: ClusterEnv,
                                train_workload: Workload = None):
        evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
            self.evaluator_method, self.parallel_evaluator)

        # Get initial solutions
        initial_sols = self.enumerate_group_configs_uneven(cluster_env)

        if self.parallel_initial_placement:
            func = ray.remote(replica_placement_fast_greedy_llm).remote
            for i in range(len(initial_sols)):
                initial_sols[i] = func(
                    initial_sols[i], model_datas, cluster_env, train_workload, None,
                    self.verbose)
            initial_sols = ray.get(initial_sols)
        else:
            for i in range(len(initial_sols)):
                initial_sols[i] = replica_placement_fast_greedy_llm(
                    initial_sols[i], model_datas, cluster_env, train_workload, evaluator,
                    self.verbose)
                #initial_sols[i] = replica_placement_beam_search(
                #    initial_sols[i], model_datas, cluster_env, train_workload, evaluator,
                #     self.beam_size, self.verbose)

        scores = evaluator.get_scores(initial_sols)
        best_idx = np.argmax(scores)
        best_sol = initial_sols[best_idx]

        return best_sol, {}


    def enumerate_separations(self,
                              model_datas: List[ModelData],
                              cluster_env: ClusterEnv):
        same_model_threshold = 0.38

        model_id_map = {}
        eco_model_datas = []
        cluster_latencies = []
        for model_id, model_data in enumerate(model_datas):
            cur_latency = max(model_data.profiling_result. \
                          para_dict[ParallelConfig(1, 1, 1)].latency[1])
            flag = False
            for i, cluster in enumerate(eco_model_datas):
                cluster_latency = max(cluster[0].profiling_result. \
                                  para_dict[ParallelConfig(1, 1, 1)].latency[1])
                if math.fabs(cur_latency - cluster_latency) / cluster_latency < same_model_threshold:
                    model_id_map[(i, len(cluster))] = model_id
                    cluster.append(model_data)
                    flag = True
                    break
            if not flag:
                model_id_map[(len(eco_model_datas), 0)] = model_id
                eco_model_datas.append([model_data])
                cluster_latencies.append(cur_latency)

        # List[List[(List[ModelData], ClusterEnv)]]
        partitions = get_partitions(cluster_env.num_devices, len(eco_model_datas))

        ## reduce num partitions
        ratio = np.empty(len(eco_model_datas), dtype=np.float32)
        for i, eco_model_data in enumerate(eco_model_datas):
            ratio[i] = sum(x.rate for x in eco_model_data)
        ratio = ratio / np.sum(ratio)   # q/s

        for threshold in [1.0, 0.5, 0.3, 0.2, 0.1]:
            reduced_partitions = []
            for partition in partitions:
                throughputs = [x / l for x, l in zip(partition, cluster_latencies)]   # q/s
                norm_throughputs = np.array(throughputs) / sum(throughputs)
                dis = np.max(np.abs(ratio - norm_throughputs))
                if dis < threshold:
                    reduced_partitions.append(partition)

            if len(reduced_partitions) < 100:
                break

        print(f"original: {len(partitions)}  reduced: {len(reduced_partitions)}")

        separations = [[(eco_model_datas[i], ClusterEnv(device_cnt, cluster_env.mem_budget)) \
                        for i, device_cnt in enumerate(partition)] \
                       for partition in reduced_partitions]

        return separations, model_id_map


    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        best_sol, _ = self.solve_placement_one_eco(model_datas, cluster_env, train_workload)

        # Separate unequal model
        if self.use_separation:
            eco_separations, model_id_map = self.enumerate_separations(model_datas, cluster_env)
            print("number of combinations: ", len(eco_separations))

            parallel = False
            if parallel:
                func = ray.remote(solve_separation_placement).remote
            else:
                func = solve_separation_placement

            sols = []
            for eco_separation in eco_separations:
                sols.append(func(self, eco_separation, model_id_map, train_workload))

            if parallel:
                sols = ray.get(sols)

            evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                self.evaluator_method, self.parallel_evaluator)
            scores = evaluator.get_scores(sols)
            best_idx = np.argmax(scores)

            evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                self.evaluator_method, self.parallel_evaluator)
            score_mixed = evaluator.get_scores([best_sol])[0]

            print(f"score_mixed: {score_mixed:.3f}, score_separate: {scores[best_idx]:.3f}")
            if scores[best_idx] > score_mixed:
                best_sol = sols[best_idx]

        if self.use_evo_search:
            best_sol = evolutionary_search(
                [best_sol], model_datas, cluster_env,
                evaluator, 200, self.verbose)
        return best_sol, {}


    # todo: may change for llm
    def enumerate_group_configs_uneven(self, cluster_env: ClusterEnv):
        sols = []
        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        for group_size in get2tok(num_devices):
            if group_size > num_devices_per_node and group_size % num_devices_per_node != 0:
                continue
            num_reg_groups = num_devices // group_size
            quo_groups = decompose2tok(num_devices % group_size)

            for pp in get_factors(group_size):
                op = group_size // pp

                if pp > self.max_pp or op > self.max_op:
                    continue

                sols.append(ModelPlacement([ParallelConfig(1, op, pp)] * num_reg_groups +
                                           [ParallelConfig(1, 1, s) for s in quo_groups],
                                           [[] for _ in range(num_reg_groups + len(quo_groups))]))
        return sols


    def enumerate_group_configs(self, cluster_env):
        sols = []
        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        for group_size in get_factors(num_devices):
            if group_size > num_devices_per_node and group_size % num_devices_per_node != 0:
                continue

            for pp in get_factors(group_size):
                op = group_size // pp
                num_groups = num_devices // group_size

                if pp > self.max_pp or op > self.max_op:
                    continue

                sols.append(ModelPlacement([ParallelConfig(1, op, pp)] * num_groups,
                                           [[] for _ in range(num_groups)]))
        return sols

    def greedy_group_configs(self,
                             model_datas: List[ModelData],
                             cluster_env: ClusterEnv,
                             train_workload: Workload,
                             evaluator: PlacementEvaluator,
                             beam_size = 3):

        assert beam_size >= 1, "beam size should >= 1."

        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        beam_sols = [[ModelPlacement([], [])]]

        for cur_num in range(1, num_devices + 1):
            ## solve sols[cur_num]
            next_sols = []
            for last_group_size in range(1, (cur_num - 1) % num_devices_per_node + 1 + 1):
                ## solve from sols[cur_num - last_group_size]
                # print("last_group_size ", last_group_size)
                for pp in get_factors(last_group_size):
                    op = last_group_size // pp
                    if pp > self.max_pp or op > self.max_op:
                        continue

                    for sol in beam_sols[cur_num - last_group_size]:
                        pre_sol = sol.copy()
                        pre_sol.group_configs.append(ParallelConfig(1, op, pp))
                        pre_sol.group_models = [[] for _ in range(len(pre_sol.group_configs))]

                        #new_sol = replica_placement_on_last_group(
                        #new_sol = replica_placement_beam_search(
                        #              pre_sol, model_datas, cluster_env, train_workload,
                        #              evaluator, self.beam_size, self.verbose)
                        new_sol = replica_placement_fast_greedy(
                                      pre_sol, model_datas, cluster_env, train_workload,
                                      evaluator, self.verbose)
 
                        next_sols.append(new_sol)
            scores = evaluator.get_scores(next_sols)
            next_indices = np.argsort(scores)[::-1][:beam_size]
            beam_sols.append([])
            for i in range(len(next_indices)):
                beam_sols[cur_num].append(next_sols[next_indices[i]])

        return beam_sols[num_devices]


class AlpaserveLLMReplacement(BasePlacementPolicy):
    
    def __init__(self,
                 replacement_interval: int,
                 max_bs: int = 1,
                 max_pp: int = 8,
                 max_op: int = 4,
                 use_evo_search: bool = False,
                 use_separation: bool = False,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_bs = max_bs
        self.max_pp = max_pp
        self.max_op = max_op
        self.n_iter = 1
        self.seed = 0
        self.beam_size = 3
        self.use_evo_search = use_evo_search
        self.use_separation = use_separation
        self.replacement_interval = replacement_interval

        self.evaluator_method = "fast_simulator"
        self.parallel_evaluator = False
        self.parallel_initial_placement = False

        if ((self.parallel_evaluator or self.parallel_initial_placement)
            and not ray.is_initialized()):
            ray.init(address="auto", ignore_reinit_error=True)


    def solve_placement_one_eco(self,
                                model_datas: List[ModelData],
                                cluster_env: ClusterEnv,
                                train_workload: Workload = None):
        evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
            self.evaluator_method, self.parallel_evaluator)

        # Get initial solutions
        initial_sols = self.enumerate_group_configs_uneven(cluster_env)

        if self.parallel_initial_placement:
            func = ray.remote(replica_placement_fast_greedy_llm).remote
            for i in range(len(initial_sols)):
                initial_sols[i] = func(
                    initial_sols[i], model_datas, cluster_env, train_workload, None,
                    self.verbose)
            initial_sols = ray.get(initial_sols)
        else:
            for i in range(len(initial_sols)):
                initial_sols[i] = replica_placement_fast_greedy_llm(
                    initial_sols[i], model_datas, cluster_env, train_workload, evaluator,
                    self.verbose)
                #initial_sols[i] = replica_placement_beam_search(
                #    initial_sols[i], model_datas, cluster_env, train_workload, evaluator,
                #     self.beam_size, self.verbose)

        scores = evaluator.get_scores(initial_sols)
        best_idx = np.argmax(scores)
        best_sol = initial_sols[best_idx]

        return best_sol, {}


    def enumerate_separations(self,
                              model_datas: List[ModelData],
                              cluster_env: ClusterEnv):
        same_model_threshold = 0.38

        model_id_map = {}
        eco_model_datas = []
        cluster_latencies = []
        for model_id, model_data in enumerate(model_datas):
            cur_latency = max(model_data.profiling_result. \
                          para_dict[ParallelConfig(1, 1, 1)].latency[1])
            flag = False
            for i, cluster in enumerate(eco_model_datas):
                cluster_latency = max(cluster[0].profiling_result. \
                                  para_dict[ParallelConfig(1, 1, 1)].latency[1])
                if math.fabs(cur_latency - cluster_latency) / cluster_latency < same_model_threshold:
                    model_id_map[(i, len(cluster))] = model_id
                    cluster.append(model_data)
                    flag = True
                    break
            if not flag:
                model_id_map[(len(eco_model_datas), 0)] = model_id
                eco_model_datas.append([model_data])
                cluster_latencies.append(cur_latency)

        # List[List[(List[ModelData], ClusterEnv)]]
        partitions = get_partitions(cluster_env.num_devices, len(eco_model_datas))

        ## reduce num partitions
        ratio = np.empty(len(eco_model_datas), dtype=np.float32)
        for i, eco_model_data in enumerate(eco_model_datas):
            ratio[i] = sum(x.rate for x in eco_model_data)
        ratio = ratio / np.sum(ratio)   # q/s

        for threshold in [1.0, 0.5, 0.3, 0.2, 0.1]:
            reduced_partitions = []
            for partition in partitions:
                throughputs = [x / l for x, l in zip(partition, cluster_latencies)]   # q/s
                norm_throughputs = np.array(throughputs) / sum(throughputs)
                dis = np.max(np.abs(ratio - norm_throughputs))
                if dis < threshold:
                    reduced_partitions.append(partition)

            if len(reduced_partitions) < 100:
                break

        print(f"original: {len(partitions)}  reduced: {len(reduced_partitions)}")

        separations = [[(eco_model_datas[i], ClusterEnv(device_cnt, cluster_env.mem_budget)) \
                        for i, device_cnt in enumerate(partition)] \
                       for partition in reduced_partitions]

        return separations, model_id_map


    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)
        
        ws = train_workload.split_time_interval(self.replacement_interval)
        
        start_times = []
        placements = []
        for i in range(len(ws)):

            best_sol, _ = self.solve_placement_one_eco(model_datas, cluster_env, ws[i])

            # Separate unequal model, default false
            if self.use_separation:
                eco_separations, model_id_map = self.enumerate_separations(model_datas, cluster_env)
                print("number of combinations: ", len(eco_separations))

                parallel = False
                if parallel:
                    func = ray.remote(solve_separation_placement).remote
                else:
                    func = solve_separation_placement

                sols = []
                for eco_separation in eco_separations:
                    sols.append(func(self, eco_separation, model_id_map, train_workload))

                if parallel:
                    sols = ray.get(sols)

                evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                    self.evaluator_method, self.parallel_evaluator)
                scores = evaluator.get_scores(sols)
                best_idx = np.argmax(scores)

                evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                    self.evaluator_method, self.parallel_evaluator)
                score_mixed = evaluator.get_scores([best_sol])[0]

                print(f"score_mixed: {score_mixed:.3f}, score_separate: {scores[best_idx]:.3f}")
                if scores[best_idx] > score_mixed:
                    best_sol = sols[best_idx]
            
            start_times.append(ws[i].arrivals[0])
            placements.append(best_sol)

        return ModelPlacementWithReplacement(start_times, placements), None


    # todo: may change for llm
    def enumerate_group_configs_uneven(self, cluster_env: ClusterEnv):
        sols = []
        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        for group_size in get2tok(num_devices):
            if group_size > num_devices_per_node and group_size % num_devices_per_node != 0:
                continue
            num_reg_groups = num_devices // group_size
            quo_groups = decompose2tok(num_devices % group_size)

            for pp in get_factors(group_size):
                op = group_size // pp

                if pp > self.max_pp or op > self.max_op:
                    continue

                sols.append(ModelPlacement([ParallelConfig(1, op, pp)] * num_reg_groups +
                                           [ParallelConfig(1, 1, s) for s in quo_groups],
                                           [[] for _ in range(num_reg_groups + len(quo_groups))]))
        return sols


    def enumerate_group_configs(self, cluster_env):
        sols = []
        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        for group_size in get_factors(num_devices):
            if group_size > num_devices_per_node and group_size % num_devices_per_node != 0:
                continue

            for pp in get_factors(group_size):
                op = group_size // pp
                num_groups = num_devices // group_size

                if pp > self.max_pp or op > self.max_op:
                    continue

                sols.append(ModelPlacement([ParallelConfig(1, op, pp)] * num_groups,
                                           [[] for _ in range(num_groups)]))
        return sols

    def greedy_group_configs(self,
                             model_datas: List[ModelData],
                             cluster_env: ClusterEnv,
                             train_workload: Workload,
                             evaluator: PlacementEvaluator,
                             beam_size = 3):

        assert beam_size >= 1, "beam size should >= 1."

        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        beam_sols = [[ModelPlacement([], [])]]

        for cur_num in range(1, num_devices + 1):
            ## solve sols[cur_num]
            next_sols = []
            for last_group_size in range(1, (cur_num - 1) % num_devices_per_node + 1 + 1):
                ## solve from sols[cur_num - last_group_size]
                # print("last_group_size ", last_group_size)
                for pp in get_factors(last_group_size):
                    op = last_group_size // pp
                    if pp > self.max_pp or op > self.max_op:
                        continue

                    for sol in beam_sols[cur_num - last_group_size]:
                        pre_sol = sol.copy()
                        pre_sol.group_configs.append(ParallelConfig(1, op, pp))
                        pre_sol.group_models = [[] for _ in range(len(pre_sol.group_configs))]

                        #new_sol = replica_placement_on_last_group(
                        #new_sol = replica_placement_beam_search(
                        #              pre_sol, model_datas, cluster_env, train_workload,
                        #              evaluator, self.beam_size, self.verbose)
                        new_sol = replica_placement_fast_greedy(
                                      pre_sol, model_datas, cluster_env, train_workload,
                                      evaluator, self.verbose)
 
                        next_sols.append(new_sol)
            scores = evaluator.get_scores(next_sols)
            next_indices = np.argsort(scores)[::-1][:beam_size]
            beam_sols.append([])
            for i in range(len(next_indices)):
                beam_sols[cur_num].append(next_sols[next_indices[i]])

        return beam_sols[num_devices]
