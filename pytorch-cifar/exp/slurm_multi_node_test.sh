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
node1="SG-IDC1-10-51-2-$4"
node2="SG-IDC1-10-51-2-$5"

#export ADAPTDL_CHECKPOINT_PATH=cifar-checkpoint
export ADAPTDL_SHARE_PATH=data
export ADAPTDL_JOB_ID=$SLURM_JOB_ID
export ADAPTDL_MASTER_ADDR=$node1
export ADAPTDL_MASTER_PORT=47016
export ADAPTDL_NUM_REPLICAS=$worker_num

for ((  i=0; i<8; i++ ))
do

  if [[ $i -lt `expr $worker_num-1` ]]
  then
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node1 python3 -u test_main.py --model=$model --bs=$batch_size --model=$model &
  else
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node1 python3 -u test_main.py --model=$model --bs=$batch_size --model=$model
  fi
done

for ((  i=8; i<16; i++ ))
do
  # node=${nodes_array[$i]}
  node=${nodes_array[0]}
  if [[ $i -lt `expr $worker_num-1` ]]
  then
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node2 python3 -u test_main.py --model=$model --bs=$batch_size --model=$model &
  else
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node2 python3 -u test_main.py --model=$model --bs=$batch_size --model=$model
  fi
done

