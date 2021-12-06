# ResNet50 host memory is full 
set -e 
for arch in PNASNetB ResNet50  GoogLeNet DenseNet121 ResNet18 MobileNetV2 # PNASNetB  # DenseNet121 ResNet18  # ResNet18 # VGG19 MobileNetV2  GoogLeNet # ResNet18 DenseNet121 ResNet18 
do 
srun --nodes=1 --gres=gpu:0 --ntasks=1 -w SG-IDC1-10-51-2-75 python -u test.py > test.log
bash slurm_test_cache.sh 1 $arch 128 75 
srun --nodes=1 --gres=gpu:0 --ntasks=1 -w SG-IDC1-10-51-2-75 python -u test.py > test.log
bash slurm_test_cache.sh 4 $arch 256 75
done 
