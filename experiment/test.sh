#--tensor-parallel-size 2

# ratescales=(1e-4 5e-4 1e-3 2e-3)
ratescales=(1e-4)
policies=(fcfs emlfq) #interleave sjmlfq 
for i in ${ratescales[*]};
do
    for j in ${policies[*]};
    do
        echo rate scale $i, policy $j
        CUDA_VISIBLE_DEVICES=1 python test.py --model /users/zyh/models/llama2-7B/ --workload-type maf1 --rate-scale $i --policy $j >output/test_${i}_${j}.out
    done
done