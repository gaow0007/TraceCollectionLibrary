from numpy import histogram
import os, sys 
sys.path.insert(0, './')
from models import * 
import numpy as np 

def collect_atomic_layer_num(net):
    atomic_layer_num = 0
    for module in net.modules():
        if isAtomicLayer(module):
            atomic_layer_num += 1
    return atomic_layer_num


def PipeTransformerMethod(layer_num, alpha, epoch):
    second_term = 0.0
    for e in range(2, epoch + 1):
        second_term += ((layer_num * alpha) / pow(1 - alpha, e))
    return pow(1 - alpha, epoch) * ((layer_num * alpha) / (1 - alpha) + second_term)


def cal_frozen_layer(epoch, tot_epoch, tot_layer_num):
    fronzen_layer_num = int(PipeTransformerMethod(tot_layer_num, 1.0 / tot_epoch, epoch)) // 2 * 2 
    return fronzen_layer_num


def isAtomicLayer(mod):
    return (isinstance(mod, nn.Conv2d))  or isinstance(mod, nn.Linear) or isinstance(mod, nn.BatchNorm2d) # and mod.in_channels != 3


def info_loader(): 
    for a1 in range(5): 
        for a2 in range(5): 
            for a3 in range(5): 
                for a4 in range(5): 
                    yield [a1, a2, a3, a4]

def isLegal(placement): 
    for i in range(3): 
        if placement[i] < placement[i+1]: 
            return False
    return True 


def notFound(ident, filename_list): 
    for filename in filename_list: 
        if ident in filename: 
            return False 
    return True 


def notFullFound(arch, placement, arch_info, filename_list): 
    for (batch, frozen) in arch_info: 
        ident = 'model_{}_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(arch, placement, batch, placement, frozen)
        if ident not in filename_list: 
            print(ident)
            # import pdb; pdb.set_trace() 
            return True 
    return False 


arch_list = ['ResNet18', 'GoogLeNet', 'VGG19', 'MobileNetV2', 'ResNet50'] 
max_list = [1024, 513, 1024, 513, 513]
local_list = [32, 45, 64, 91, 129, 182, 257, 363, 513, 725, 1024]

node_list = ['75', '77', '67', '30'] 
history_list = list() 

filename_list = list() 
for filename in os.listdir('speed'):     
    filename_list.append(filename)

arch_info = dict() 
for arch in arch_list: 
    arch_info[arch] = list() 
    """
    for filename in filename_list: 
        batch = int(filename.split('bs_')[1].split('_placement')[0])
        frozen = int(filename.split('frozen_')[1].split('.npy')[0])
        if (batch, frozen) not in arch_info[arch]: 
            arch_info[arch].append((batch, frozen)) 
            # print(arch, (batch, frozen))
    """
    if arch != 'VGG19':
        net = eval(arch)()
    else:
        net = VGG('VGG19') 
    tot_layer_num = collect_atomic_layer_num(net)
    profile_tot_layer = int(tot_layer_num * 0.9) # // 2
    arch_info[arch] = sorted(arch_info[arch])
    frozen_list = np.linspace(0, profile_tot_layer, 10)
    frozen_list = [int(layer) for layer in frozen_list] 
    for batch in local_list: 
        if batch > max_list[arch_list.index(arch)]: continue 
        for frozen in frozen_list: 
            # print(arch, (batch, frozen))
            arch_info[arch].append((batch, frozen)) 

# exit(0)


print('set -e')
cnt = 0
for placement in info_loader(): 
    if isLegal(placement): 
        while 0 in placement: 
            placement.remove(0) 
        if len(placement) <= 0: 
            continue 
        placement_str = '-'.join([str(pm) for pm in placement])
        if placement_str in history_list: 
            continue 

        history_list.append(placement_str)

        for arch, max_bs in zip(arch_list, max_list): 
            # bash exp/slurm_run_speed.sh 1 $arch 1 $node 
            node_str = "'{}'".format(' '.join(node_list[:len(placement)]))
            placement_str = [str(pm) for pm in placement]
            cmd = 'bash exp/speed/slurm_run_auto_speed_{}.sh {} {} {} {} {}'.format(cnt%2, sum(placement), arch, "'{}'".format(' '.join(placement_str)), node_str, max_bs)
            ident = 'model_{}_placement_{}_'.format(arch, '-'.join(placement_str))
            # if notFound(ident, filename_list): 
            if notFullFound(arch, '-'.join(placement_str), arch_info[arch], filename_list): 
                # print(ident)
                # continue 
                print('''echo "{}" '''.format(cmd))
                if cnt % 2 == 0: 
                    print(cmd+'&') 
                    # print(cmd)
                else: 
                    print(cmd)
                cnt += 1
            
        # break 
# python exp/speed_run_fill.py > exp/placement_speed.sh 
