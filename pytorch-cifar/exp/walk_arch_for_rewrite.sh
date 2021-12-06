# ResNet50 host memory is full 
set -e 
for arch in MobileNetV2 ResNet50 # ResNet18 #   GoogLeNet MobileNetV2  PNASNetB  # GoogLeNet  ResNet50 # ResNet18 DenseNet121 ResNet50 MobileNetV2  PNASNetB GoogLeNet # PNASNetB  # DenseNet121 ResNet18 DenseNet121 # ResNet18 # VGG19 MobileNetV2  GoogLeNet #
do 
# srun --nodes=1 --gres=gpu:0 --ntasks=1 -w SG-IDC1-10-51-2-75 python -u test.py > test.log
# bash slurm_test_rewrite.sh 1 $arch 128 75 
# srun --nodes=1 --gres=gpu:0 --ntasks=1 -w SG-IDC1-10-51-2-75 python -u test.py > test.log
bash slurm_test_rewrite.sh 4 $arch 512 75 > rewrite/$arch.log 
bash slurm_test_no_rewrite.sh 4 $arch 512 75 > rewrite/no_$arch.log
done 
