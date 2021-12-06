#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=2
#SBATCH --tasks-per-node 1
#SBATCH --partition dsta*

worker_num=$1
model=$2
placement=$3
last_rank=`expr $worker_num - 1`

# nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes="SG-IDC1-10-51-2-$4"
nodes_array=( $nodes )
node1=${nodes_array[0]}
# mkdir -p ./cifar_ckpt/$model/$worker_num/
# export ADAPTDL_CHECKPOINT_PATH=./cifar_ckpt/$model/$worker_num/
export ADAPTDL_SHARE_PATH=data
export ADAPTDL_JOB_ID=$SLURM_JOB_ID
export ADAPTDL_MASTER_ADDR=$node1
export ADAPTDL_MASTER_PORT=42011
export ADAPTDL_NUM_REPLICAS=$worker_num
export TARGET_BATCH_SIZE=$batch_size 

# batch_size=128
batch_size=128 

for ((  i=0; i<$worker_num; i++ ))
do
  # node=${nodes_array[$i]}
  node=${nodes_array[0]}
  if [[ $i -lt `expr $worker_num-1` ]]
  then
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node python3 -u main_speed.py --model=$model --bs=$batch_size --placement=$placement &
  else
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node python3 -u main_speed.py --model=$model --bs=$batch_size --placement=$placement
  fi
done