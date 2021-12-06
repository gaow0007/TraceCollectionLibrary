import numpy as np 

info = np.load('stats/model_ResNet18_bs_128_frozen_False', allow_pickle=True).tolist() 
# for i in range(10): 
info_list = [info['layer_{}_grad_var'.format(i)][0] for i in range(40)]
print(info_list)
print(sum(info_list))
