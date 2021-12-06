#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=2
#SBATCH --tasks-per-node 1
#SBATCH --partition dsta*

worker_num=$1
dataset=$2
placement=($3)
nodes=($4)
max_bs=$5

export ADAPTDL_SHARE_PATH=data
export ADAPTDL_JOB_ID=$SLURM_JOB_ID
export ADAPTDL_MASTER_ADDR='SG-IDC1-10-51-2-'${nodes[0]}
export ADAPTDL_MASTER_PORT=47613
export ADAPTDL_NUM_REPLICAS=$worker_num
# export TARGET_BATCH_SIZE=$batch_size 

# batch_size=128
batch_size=128 

length=${#placement[@]}

base=0
for (( node=0; node<$length; node++ ))
do 
    # echo ${nodes[$node]}
    node_id='SG-IDC1-10-51-2-'${nodes[$node]}
    gpu_num=${placement[$node]} 
    placement_str=$(IFS=- ; echo "${placement[*]}") 
    left=$base 
    right=$(($left+$gpu_num)) 
    for (( i=$left; i<$right; i++ )) 
    do
        if [[ $i -lt `expr $worker_num-1` ]]
        then 
            # echo ${placement[@]}
            ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node_id python3 train_speed.py  --weight_path=./weight/darknet53_448.weights --placement=$placement_str &
        else
            # echo "$i"_"$node_id"
            ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node_id python3 train_speed.py  --weight_path=./weight/darknet53_448.weights --placement=$placement_str
        fi
    done 
    base=$right 
done