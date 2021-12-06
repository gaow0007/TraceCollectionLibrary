# node=75
for arch in  GoogLeNet VGG19 MobileNetV2 ResNet50 # ResNet18
do 
    # bash exp/slurm_run_speed.sh 1 $arch 1 $node
    # bash exp/slurm_run_speed.sh 2 $arch 2 $node
    # bash exp/slurm_run_speed.sh 3 $arch 3 $node
    # bash exp/slurm_run_speed.sh 4 $arch 4 $node
    bash exp/slurm_multi_node_run.sh 16 $arch 2048 160 0.08 75 78 
    # bash slurm_4_node_run.sh 32 $arch 4096 160 0.08 65 67 75 77 
done 

# exit 


# node=$1
# for arch in $2 # ResNet18 VGG19 MobileNetV2 ResNet50 GoogLeNet
# do 
#     # bash exp/slurm_run.sh 1 $arch 128 100  0.08 $node 
#     # bash exp/slurm_run.sh 2 $arch 256 140 0.08 $node 
#     # bash exp/slurm_run.sh 4 $arch 512 160 0.10 $node  
#     bash exp/slurm_run.sh 8 $arch 1024 160 0.22 $node  

#     # bash exp/slurm_multi_node_run.sh 16 $arch 2048 160 0.08 72 79
    
# done 
exit 

for arch in ResNet18 VGG19 MobileNetV2 ResNet50 GoogLeNet
do 
    # bash exp/slurm_run.sh 1 $arch 128 100  0.08 $node 
    # bash exp/slurm_run.sh 2 $arch 256 140 0.08 $node 
    # bash exp/slurm_run.sh 4 $arch 512 160 0.10 $node  
    # bash exp/slurm_run.sh 8 $arch 1024 160 0.22 $node  
    bash exp/slurm_4_node_run.sh 32 $arch 4096 160 0.08 78 79 72 68
    
done 

exit 

# {
# node=75
# for arch in GoogLeNet VGG19  # ResNet18 GoogLeNet # VGG19  ResNet50 # ResNet18 DenseNet121 # ResNet18 # VGG19 MobileNetV2  GoogLeNet # ResNet18 DenseNet121 ResNet18
# do 
#     bash slurm_run.sh 1 $arch 128 100  0.08 $node
#     bash slurm_run.sh 2 $arch 256 140 0.08 $node  
#     bash slurm_run.sh 4 $arch 512 160 0.10 $node  
#     bash slurm_run.sh 8 $arch 1024 160 0.22 $node  
#     # bash slurm_multi_node_run.sh 16 $arch 2048 160 0.08 75 77 
#     # bash slurm_run.sh 8 $arch 4096 200 0.45 $node  
# done 
# } &


# {
# node=68
# for arch in ResNet50 MobileNetV2  # ResNet18 GoogLeNet # VGG19  ResNet50 # ResNet18 DenseNet121 # ResNet18 # VGG19 MobileNetV2  GoogLeNet # ResNet18 DenseNet121 ResNet18
# do 
#     bash slurm_run.sh 1 $arch 128 100  0.08 $node
#     bash slurm_run.sh 2 $arch 256 140 0.08 $node  
#     bash slurm_run.sh 4 $arch 512 160 0.10 $node  
#     bash slurm_run.sh 8 $arch 1024 160 0.22 $node  
#     # bash slurm_multi_node_run.sh 16 $arch 2048 160 0.08 75 77 
#     # bash slurm_run.sh 8 $arch 4096 200 0.45 $node  
# done 
# } &


# node1=77
# node2=78
# for arch in  ResNet18 ResNet50 MobileNetV2 DenseNet121 VGG19 ShuffleNetV2 ResNeXt29_2x64d ResNet101 PNASNetB GoogLeNet
# do 
#  bash slurm_multi_node_test.sh 16 $arch 2048 $node1 $node2
#  sleep 5
#  bash slurm_multi_node_test.sh 16 $arch 4096 $node1 $node2
#  sleep 5
#  bash slurm_multi_node_test.sh 16 $arch 1024 $node1 $node2
#  sleep 5
# done

# exit 


# VGG19 PNASNetB ShuffleNetV2 ResNet50
# for arch in  ResNeXt29_2x64d ResNet101 PNASNetB  # ResNet18 MobileNetV2 DenseNet121 
# do 
#  bash slurm_test.sh 8 $arch 512 $node
#  bash slurm_test.sh 8 $arch 1024 $node
#  bash slurm_test.sh 8 $arch 2048 $node
# done 

# exit 
# {
# arch=MobileNetV2
# bash slurm_test.sh 2 $arch 128 70
# sleep 5
# bash slurm_test.sh 2 $arch 256 70
# sleep 5
# bash slurm_test.sh 2 $arch 512 70
# sleep 5
# bash slurm_test.sh 4 $arch 128 70
# sleep 5
# bash slurm_test.sh 4 $arch 256 70
# sleep 5
# bash slurm_test.sh 4 $arch 512 70
# sleep 5
# } &

# {
# node=78

# arch=ShuffleNetV2
# # bash slurm_test.sh 2 $arch 512 $node
# # sleep 5
# bash slurm_test.sh 2 $arch 128 $node
# sleep 5
# bash slurm_test.sh 2 $arch 256 $node
# sleep 5

# bash slurm_test.sh 4 $arch 128 $node
# sleep 5
# bash slurm_test.sh 4 $arch 256 $node
# sleep 5
# bash slurm_test.sh 4 $arch 512 $node
# sleep 5
# }  & 

# node=76
# arch=GoogLeNet # ResNet101
# # arch=GoogLeNet
# bash slurm_test.sh 2 $arch 128 $node
# sleep 5
# bash slurm_test.sh 2 $arch 256 $node
# sleep 5
# bash slurm_test.sh 2 $arch 512 $node
# sleep 5
# bash slurm_test.sh 4 $arch 128 $node
# sleep 5
# bash slurm_test.sh 4 $arch 256 $node
# sleep 5
# bash slurm_test.sh 4 $arch 512 $node
# sleep 5

exit 

arch=DenseNet121
bash slurm_test.sh 2 $arch 128
sleep 5
bash slurm_test.sh 2 $arch 256
sleep 5
bash slurm_test.sh 2 $arch 512
sleep 5
bash slurm_test.sh 4 $arch 128
sleep 5
bash slurm_test.sh 4 $arch 256
sleep 5
bash slurm_test.sh 4 $arch 512
sleep 5
exit 



bash slurm_run.sh 2 DenseNet121
sleep 5
bash slurm_run_frozen.sh 2 DenseNet121
sleep 5

bash slurm_run.sh 2 ResNet18
sleep 5
bash slurm_run_frozen.sh 2 ResNet18
sleep 5



bash slurm_run.sh 2 MobileNetV2
sleep 5
bash slurm_run_frozen.sh 2 MobileNetV2
sleep 5

# bash slurm_test.sh $node_num
arch=ResNet18
bash slurm_test.sh 2 $arch 128
sleep 5

bash slurm_test.sh 2 $arch 256
sleep 5
bash slurm_test.sh 2 $arch 512
sleep 5
bash slurm_test.sh 4 $arch 128
sleep 5
bash slurm_test.sh 4 $arch 256
sleep 5
bash slurm_test.sh 4 $arch 512
sleep 5

arch=DenseNet121
bash slurm_test.sh 2 $arch 128
sleep 5
bash slurm_test.sh 2 $arch 256
sleep 5
bash slurm_test.sh 2 $arch 512
sleep 5

bash slurm_test.sh 4 $arch 128
sleep 5
bash slurm_test.sh 4 $arch 256
sleep 5
bash slurm_test.sh 4 $arch 512
sleep 5


arch=MobileNetV2
bash slurm_test.sh 2 $arch 128
sleep 5
bash slurm_test.sh 2 $arch 256
sleep 5
bash slurm_test.sh 2 $arch 512
sleep 5

bash slurm_test.sh 4 $arch 128
sleep 5
bash slurm_test.sh 4 $arch 256
sleep 5
bash slurm_test.sh 4 $arch 512
sleep 5
