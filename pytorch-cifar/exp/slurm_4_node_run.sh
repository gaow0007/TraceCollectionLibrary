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
epoch=$4
lr=$5

last_rank=`expr $worker_num - 1`

# nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
node1="SG-IDC1-10-51-2-$6"
node2="SG-IDC1-10-51-2-$7"
node3="SG-IDC1-10-51-2-$8"
node4="SG-IDC1-10-51-2-$9"

#export ADAPTDL_CHECKPOINT_PATH=cifar-checkpoint
export ADAPTDL_SHARE_PATH=data
export ADAPTDL_JOB_ID=$SLURM_JOB_ID
export ADAPTDL_MASTER_ADDR=$node1
export ADAPTDL_MASTER_PORT=47026
export ADAPTDL_NUM_REPLICAS=$worker_num
export TARGET_BATCH_SIZE=$batch_size 
mkdir -p ./cifar_ckpt/$model/$worker_num/
export ADAPTDL_CHECKPOINT_PATH=./cifar_ckpt/$model/$worker_num/

batch_size=128 

for ((  i=0; i<8; i++ ))
do
  if [[ $i -lt `expr $worker_num-1` ]]
  then
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node1 python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz --epoch=$epoch --lr=$lr &
  else
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node1 python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz --epoch=$epoch --lr=$lr &
  fi
done

for ((  i=8; i<16; i++ ))
do
  # node=${nodes_array[$i]}
  node=${nodes_array[0]}
  if [[ $i -lt `expr $worker_num-1` ]]
  then
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node2 python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz --epoch=$epoch --lr=$lr &
  else
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node2 python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz --epoch=$epoch --lr=$lr
  fi
done


for ((  i=16; i<24; i++ ))
do
  # node=${nodes_array[$i]}
  node=${nodes_array[0]}
  if [[ $i -lt `expr $worker_num-1` ]]
  then
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node3 python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz --epoch=$epoch --lr=$lr &
  else
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node3 python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz --epoch=$epoch --lr=$lr
  fi
done

for ((  i=24; i<32; i++ ))
do
  # node=${nodes_array[$i]}
  node=${nodes_array[0]}
  if [[ $i -lt `expr $worker_num-1` ]]
  then
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node4 python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz --epoch=$epoch --lr=$lr &
  else
    ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node4 python3 -u main.py --model=$model --bs=$batch_size --autoscale-bsz --epoch=$epoch --lr=$lr
  fi
done
