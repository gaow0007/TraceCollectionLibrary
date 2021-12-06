import numpy as np 

info = np.load('stats/model_WikiText2_bs_4096', allow_pickle=True).tolist() 
# for i in range(10): 
info_list = [info['layer_{}_grad_var'.format(i)][0] for i in range(70)]
print(info_list)
print(sum(info_list))
