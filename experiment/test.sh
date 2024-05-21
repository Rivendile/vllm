python test.py --model /users/zyh/models/llama2-7B/ --workload-type maf1 --rate-scale 1e-4 --policy fcfs >test.out
#--tensor-parallel-size 2

# CUDA_VISIBLE_DEVICES=0 python test.py --model /users/zyh/models/llama2-7B/ --workload-type maf1 --rate-scale 5e-4 --policy fcfs >test5e-4.out

# CUDA_VISIBLE_DEVICES=0 python test.py --model /users/zyh/models/llama2-7B/ --workload-type maf1 --rate-scale 1e-3 --policy fcfs >test1e-3.out

# CUDA_VISIBLE_DEVICES=0 python test.py --model /users/zyh/models/llama2-7B/ --workload-type maf1 --rate-scale 2e-3 --policy fcfs >test2e-3.out
