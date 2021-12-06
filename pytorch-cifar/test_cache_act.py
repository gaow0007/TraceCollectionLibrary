import torch 
from adaptdl.torch.cache.act_cache import ActCache
import time 
dataset_sample_size=10000

act_cache = ActCache(True, dataset_sample_size=10000)

local_batch_size = 101 

last_idx = 0
data_iter_count = (dataset_sample_size - 1) // local_batch_size + 1
for i in range(data_iter_count): 
    batch_size = min(local_batch_size, dataset_sample_size - i * local_batch_size)
    if batch_size < 0: 
        continue 
    batch_sample_idx = [last_idx + j for j in range(batch_size)]
    last_idx = max(batch_sample_idx) + 1

    output = torch.randn(batch_size, 64, 32, 32)
    act_cache.save_cache_feature(10, batch_sample_idx, output.detach().cpu())

act_cache.update_num_frozen_layer(10)


start_time = time.time() 
batch_size = 100
while True: 
    time.sleep(1)
    print(act_cache.get_cache_sampe_size(10))
    if act_cache.get_cache_sampe_size(10) == dataset_sample_size: 
        for i in range(data_iter_count): 
            print('processing {}'.format(i))
            batch_sample_idx = [i * batch_size + j for j in range(batch_size)] 
            output = None 
            output = act_cache.load_cache_feature(10, batch_sample_idx, output)
            print(output.shape)
        break

end_time = time.time() 
print(start_time - end_time)
print('clean up')
act_cache.cleanup()