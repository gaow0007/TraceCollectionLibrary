
node="SG-IDC1-10-51-2-75"

# srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node python3 -u graph_rewriter.py 
srun --nodes=1 --gres=gpu:1 --ntasks=1 -w $node python3 -u test_topology.py 