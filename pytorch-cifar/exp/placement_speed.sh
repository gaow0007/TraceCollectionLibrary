set -e
echo "bash exp/speed/slurm_run_auto_speed_0.sh 2 VGG19 '1 1' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_0.sh 2 VGG19 '1 1' '78 70' 1024
echo "bash exp/speed/slurm_run_auto_speed_1.sh 3 VGG19 '2 1' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_1.sh 3 VGG19 '2 1' '78 70' 1024
echo "bash exp/speed/slurm_run_auto_speed_0.sh 4 VGG19 '2 2' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_0.sh 4 VGG19 '2 2' '78 70' 1024
echo "bash exp/speed/slurm_run_auto_speed_1.sh 4 VGG19 '3 1' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_1.sh 4 VGG19 '3 1' '78 70' 1024
echo "bash exp/speed/slurm_run_auto_speed_0.sh 5 VGG19 '3 2' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_0.sh 5 VGG19 '3 2' '78 70' 1024
echo "bash exp/speed/slurm_run_auto_speed_1.sh 6 VGG19 '3 3' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_1.sh 6 VGG19 '3 3' '78 70' 1024
echo "bash exp/speed/slurm_run_auto_speed_0.sh 5 VGG19 '4 1' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_0.sh 5 VGG19 '4 1' '78 70' 1024
echo "bash exp/speed/slurm_run_auto_speed_1.sh 6 VGG19 '4 2' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_1.sh 6 VGG19 '4 2' '78 70' 1024
echo "bash exp/speed/slurm_run_auto_speed_0.sh 7 VGG19 '4 3' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_0.sh 7 VGG19 '4 3' '78 70' 1024
echo "bash exp/speed/slurm_run_auto_speed_1.sh 8 VGG19 '4 4' '78 70' 1024" 
bash exp/speed/slurm_run_auto_speed_1.sh 8 VGG19 '4 4' '78 70' 1024
