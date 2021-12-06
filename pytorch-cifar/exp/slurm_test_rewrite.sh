#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=2
#SBATCH --tasks-per-node 1
#SBATCH --partition dsta*

worker_num=$1
model=$2
batch_size=$3

last_rank=`expr $worker_num - 1`

# nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes="SG-IDC1-10-51-2-$4"
nodes_array=( $nodes )
node1=${nodes_array[0]}

#export ADAPTDL_CHECKPOINT_PATH=cifar-checkpoint
export ADAPTDL_SHARE_PATH=data
export ADAPTDL_JOB_ID=$SLURM_JOB_ID
export ADAPTDL_MASTER_ADDR=$node1
export ADAPTDL_MASTER_PORT=46050
export ADAPTDL_NUM_REPLICAS=$worker_num


for ((  i=0; i<$worker_num; i++ ))
do
  # node=${nodes_array[$i]}
  node=${nodes_array[0]}
  if [[ $i -lt `expr $worker_num-1` ]]
  then
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node python3 -u test_cache_main.py --model=$model --bs=$batch_size --model=$model --rewrite &
  else
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node python3 -u test_cache_main.py --model=$model --bs=$batch_size --model=$model --rewrite 
  fi
done

# batch_size=128
# for ((  i=0; i<$worker_num; i++ ))
# do
#   # node=${nodes_array[$i]}
#   node=${nodes_array[0]}
#   if [[ $i -lt `expr $worker_num-1` ]]
#   then
#     ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz &
#   else
#     ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz
#   fi
# done