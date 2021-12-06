node=75
# worker_num=$1 
# placement=($2)
# nodes=($3)
# batch_size=$4
for data in WikiText2 
do 
    # worker, node, bs, port, epoch 
    # bash exp/stats/slurm_run.sh 1  '1' $node 4096 46221 4 $data &
    # bash exp/stats/slurm_run.sh 2  '2' $node 8192 43221 8 $data  &
    # bash exp/stats/slurm_run.sh 4  '4' $node 16384 46321 16 $data 
    bash exp/stats/slurm_run.sh 8  '8' $node 32768 46321 32 $data & 
    bash exp/stats/slurm_run.sh 16  '8 8' '79 70' 65536 46321 64 $data 
    # bash exp/stats/slurm_run.sh 32  '8 8 8 8' '79 72 70 68' 131072 46321 128 $data
done 

# bash exp/stats/bash_stats.sh
