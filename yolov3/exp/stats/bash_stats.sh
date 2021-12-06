node=34
# worker_num=$1 
# placement=($2)
# nodes=($3)
# batch_size=$4
for arch in yolo # ResNet18 GoogLeNet # VGG19  ResNet50 # ResNet18 DenseNet121 # ResNet18 # VGG19 MobileNetV2  GoogLeNet # ResNet18 DenseNet121 ResNet18
do 
    # bash exp/stats/slurm_run.sh 1  '1' $node 8 47121 50 &
    # bash exp/stats/slurm_run.sh 2  '2' $node 16 47221 100 &
    # bash exp/stats/slurm_run.sh 4  '4' $node 32 47321 100 
    # bash exp/stats/slurm_run.sh 8  '8' $node 64 47021 100
    # bash exp/stats/slurm_run.sh 16  '8 8' '70 78' 128 47021 100
    bash exp/stats/slurm_run.sh 32  '8 8 8 8' '78 70 66 65' 256 47021 100 
done 

# bash exp/stats/bash_stats.sh
