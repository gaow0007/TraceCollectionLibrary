# placement=(2 1 2 3)

worker_num=$1
model=$2
placement=$3
placement=(2 1 2 3)
nodes=(65 67 77 55)


export ADAPTDL_SHARE_PATH=data
export ADAPTDL_JOB_ID=$SLURM_JOB_ID
export ADAPTDL_MASTER_ADDR=$node1
export ADAPTDL_MASTER_PORT=42011
export ADAPTDL_NUM_REPLICAS=$worker_num
export TARGET_BATCH_SIZE=$batch_size 

# batch_size=128
batch_size=128 

length=${#placement[@]}

base=0
for (( node=0; node<$length; node++ ))
do 
    echo ${nodes[$node]}
    node_id='SG-IDC1-10-51-2-'${nodes[$node]}
    gpu_num=${placement[$node]} 
    left=$base 
    right=$(($left+$gpu_num))

    
    # echo $right 
    for (( i=$left; i<$right; i++ )) 
    do
        if [[ $i -lt `expr $worker_num-1` ]]
        then 
            # echo "$i"_"$node"
            echo "$i"_"$node_id"
            # ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node_id python3 -u main_speed.py --model=$model --bs=$batch_size --placement=$placement &
        else
            echo "$i"_"$node_id"
            # ADAPTDL_REPLICA_RANK=$i srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node_id python3 -u main_speed.py --model=$model --bs=$batch_size --placement=$placement
        fi
    done 

    base=$right 
done