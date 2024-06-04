#--tensor-parallel-size 2

ratescales=(1e-1) #5e-4 1e-3 2e-3
# ratescales=(1e-5 2e-5 4e-5 8e-5 1e-4 2e-4 4e-4)
policies=(fcfs emlfq) #fcfs interleave sjmlfq emlfq
for i in ${ratescales[*]};
do
    for j in ${policies[*]};
    do
        echo rate scale $i, policy $j
        # python test.py --model /users/zyh/models/llama2-7B/ --workload-type maf1 --rate-scale $i --policy $j >output/test_req.out
        python test.py --model /users/zyh/models/Llama-2-13b-chat-hf/ --workload-type maf1 --rate-scale $i --policy $j --max-num-seqs 8 --strict-stop >output/test_${i}_${j}.out
        # python test.py --model /users/zyh/models/llama2-7B/ --workload-type maf1 --rate-scale $i --policy $j --tensor-parallel-size 2 >output/test_${i}_${j}_tp2.out
    done
done