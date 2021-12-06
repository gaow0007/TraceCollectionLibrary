import numpy as np 
import os 
identify = 'model_ResNet18'
root = 'speed'
for path in os.listdir(root): 
    if 'placement_1-1' not in path: continue 
    # if 'frozen_0' not in path: continue 
    if 'bs_32' not in path: continue 
    filename = os.path.join(root, path)
    info = np.load(filename, allow_pickle=True).tolist() 
    for key, value in info.items(): 
        if 'metric_' in key: 
            profile = value[0].profile 
            for k, v in profile.items(): 
                print(filename)
                print('step time {}'.format(v['optim_step_time'] / v['optim_count']))