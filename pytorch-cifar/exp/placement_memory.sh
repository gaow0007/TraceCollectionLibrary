set -e
# echo "bash exp/speed/slurm_run_auto_speed_1.sh 6 GoogLeNet '3 2 1' '37 67 68' 513" 
# bash exp/speed/slurm_run_auto_speed_1.sh 6 GoogLeNet '3 2 1' '37 67 68' 513
# echo "bash exp/speed/slurm_run_auto_speed_0.sh 6 GoogLeNet '3 1 1 1' '37 67 68 78' 513" 
# bash exp/speed/slurm_run_auto_speed_0.sh 6 GoogLeNet '3 1 1 1' '37 67 68 78' 513

bash exp/speed/slurm_run_auto_memory_1.sh 1 GoogLeNet '1' '75' 725 & 
bash exp/speed/slurm_run_auto_memory_1.sh 1 ResNet18 '1' '68' 725 & 
bash exp/speed/slurm_run_auto_memory_1.sh 1 VGG19 '1' '34' 725 & 
bash exp/speed/slurm_run_auto_memory_1.sh 1 MobileNetV2 '1' '37' 725 & 
bash exp/speed/slurm_run_auto_memory_1.sh 1 ResNet50 '1' '77' 725 & 
