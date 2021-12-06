node=76
prefix="srun -p dsta --mpi=pmi2 --gres=gpu:8 -n8 --ntasks-per-node=1 --job-name=ttest --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-$node "

for arch in ResNet18 # VGG19 MobileNetV2 ResNet18 #MobileNetV2 #  ResNet18 MobileNetV2 VGG19 ResNet50 GoogLeNet DenseNet121   # ResNet50 # SENet18 GoogLeNet # DenseNet121
do
# $prefix 
# python -u main.py --model=$arch  --epoch=100 
# ADAPTDL_MASTER_ADDR=10.51.2.$node ADAPTDL_MASTER_PORT=9012 ADAPTDL_NUM_REPLICAS=8 ADAPTDL_REPLICA_RANK=0 srun -p dsta --mpi=pmi2 --gres=gpu:8 -n8 --job-name=ttest --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-$node  python -u main.py --model=$arch  --epoch=100 
# ADAPTDL_MASTER_ADDR=0.0.0.0 ADAPTDL_MASTER_PORT=9015 ADAPTDL_NUM_REPLICAS=2 ADAPTDL_REPLICA_RANK=0 python -u main.py --model=$arch  --epoch=100 
ADAPTDL_MASTER_ADDR=0.0.0.0 ADAPTDL_MASTER_PORT=9015 ADAPTDL_NUM_REPLICAS=2 ADAPTDL_REPLICA_RANK=0 python -u main.py --model=ResNet18  --epoch=100 
ADAPTDL_MASTER_ADDR=0.0.0.0 ADAPTDL_MASTER_PORT=9015 ADAPTDL_NUM_REPLICAS=2 ADAPTDL_REPLICA_RANK=1 python -u main.py --model=ResNet18  --epoch=100 
# ADAPTDL_MASTER_ADDR=localhost ADAPTDL_MASTER_PORT=9012 ADAPTDL_NUM_REPLICAS=2 ADAPTDL_REPLICA_RANK=1 python -u main.py --model=$arch  --epoch=100 

done