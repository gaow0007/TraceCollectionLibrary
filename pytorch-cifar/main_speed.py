# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import os 
import copy 
# os.environ['TARGET_BATCH_SIZE'] = "128"
# os.environ['TRACE_EFFICIENCY'] = "True"
import argparse
import time
import collections
import numpy as np

from models import *

import adaptdl
import adaptdl.torch as adl

from adaptdl.torch._metrics import _metrics_state, get_progress

from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter

import random
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.08, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')

parser.add_argument('--model', default='ResNet18', type=str, help='model')
parser.add_argument('--placement', required=True, type=str, help='placement')
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False,
                    action='store_true', help='autoscale batchsize')
parser.add_argument('--frozen', dest='frozen', default=False,
                    action='store_true', help='autoscale batchsize')
parser.add_argument('--mixed-precision', dest='mixed_precision', default=False,
                    action='store_true', help='use automatic mixed precision')
parser.add_argument('--max_bs', default=1024, type=int, help='number of epochs')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda', 'require GPU device'
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

if adaptdl.env.replica_rank() == 0:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = adl.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=False)
    dist.barrier()  # We use a barrier here so that non-master replicas would wait for master to download the data
else:
    dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = adl.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=False)

# if args.autoscale_bsz:
#     trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1024), gradient_accumulation=False)
# trainloader.autoscale_batch_size(args.bs, local_bsz_bounds=(128, 1024), gradient_accumulation=False) 


validset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
validloader = adl.AdaptiveDataLoader(validset, batch_size=256, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
if args.model != 'VGG19':
    net = eval(args.model)()
else:
    net = VGG('VGG19')
    
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([{"params": [param]} for param in net.parameters()],
                      lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ExponentialLR(optimizer, 0.0133 ** (1.0 / args.epochs))
# lr_scheduler = MultiStepLR(optimizer, [50, 75], 0.1)


# profile layer info 
adl.profile_layer_info(net, torch.randn(1, 3, 32, 32))
net = adl.AdaptiveDataParallel(net, optimizer, lr_scheduler, find_unused_parameters=True) 
adaptdl.torch.current_fronzen_layer() 


# Training
def train(epoch, limit=20):
    print('\nEpoch: %d' % epoch)
    net.train()
    frozen_net(net)
    stats = adl.Accumulator()
    for idx, (inputs, targets) in enumerate(trainloader):  
        if idx >= limit:
            break  
        optimizer.zero_grad() 
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward() 
        optimizer.step() 
        stats["loss_sum"] += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        stats["total"] += targets.size(0)
        stats["correct"] += predicted.eq(targets).sum().item() 


    trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data/")
    net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model/") 
    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
        # print("Train:", stats)
    return idx + 1


def valid(epoch):
    net.eval()
    stats = adl.Accumulator()
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            stats["loss_sum"] += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            stats["total"] += targets.size(0)
            stats["correct"] += predicted.eq(targets).sum().item()
    
    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Valid", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Valid", stats["accuracy"], epoch)
        return stats["accuracy"]


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

def frozen_net(net):
    for module in net.modules():
        if isAtomicLayer(module):
            frozen = True
            for param in module.parameters():
                if param.requires_grad == True:
                    frozen = False
            if frozen:
                module.eval() 

def apply_frozen(net, fronzen_layer_num): 
    for module in net.modules():
        if isAtomicLayer(module) :
            if fronzen_layer_num > 0:
                module.eval()
                fronzen_layer_num -= 1
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
        
    # remove backward hook
    import functools
    for name, param in net.named_parameters():
        if param.requires_grad == False:
            if name in net.param_hooks:
                net.param_hooks[name].remove()
                net.param_hooks.pop(name)
        else:
            if name not in net.param_hooks:
                handle = param.register_hook(functools.partial(net._backward_hook, param))
                net.param_hooks[name] = handle
    
    # remove param_groups
    for group_idx, (group, net_param) in enumerate(zip(net.adascale._optimizer.param_groups, net.parameters())):
        need_remove = False 
        if net_param.requires_grad == False:
            for param_idx, param in enumerate(group["params"]):
                if param is not None and param.requires_grad == False:
                    handle = net.adascale.gns_param_hooks[group_idx][param_idx]
                    handle.remove() 
                    need_remove = True
        
            if need_remove:
                group["params"] = list() 
        elif net_param.requires_grad == True and len(group["params"]) == 0:
            group["params"] = [net_param]

def set_local_batch_size(local_batch_size): 
    batch_size = local_batch_size * adaptdl.env.num_replicas() 
    os.environ['TARGET_BATCH_SIZE'] = "{}".format(batch_size) 

def getCurTime():
    torch.cuda.synchronize(0)
    return time.time() 

def walkover(trainloader): 
    for idx, (_, _) in enumerate(trainloader):  
        pass

tot_layer_num = collect_atomic_layer_num(net)
profile_tot_layer = int(tot_layer_num * 0.9) # // 2
print(adaptdl.env.job_id())
tensorboard_dir = 'log_dir'


from adaptdl.torch._metrics import _metrics_state

with SummaryWriter(tensorboard_dir) as writer:
    
    # for local_batch_size in [513, 725, 1024]: 
    # for local_batch_size in [513, 725, 1024]: 
    frozen_list = np.linspace(0, profile_tot_layer, 10)
    frozen_list = [int(layer) for layer in frozen_list] 
    for epoch, fronzen_layer_num in enumerate(frozen_list):
        adl.set_current_frozen_layer(fronzen_layer_num)
        apply_frozen(net, fronzen_layer_num) 
        

        for local_batch_size in [32, 45, 64, 91, 129, 182, 257, 363, 513, 725, 1024]: 
            if local_batch_size > args.max_bs: 
                continue 
            print('local_batch_size == {}'.format(local_batch_size))
            set_local_batch_size(local_batch_size=local_batch_size)
            epoch_frequency = 1
            training_time = 0
            time_list = list()
            metric_list = list() 
            metric_info = _metrics_state()
            info_dict = dict() 
            metric_info.profile = collections.defaultdict(collections.Counter) 
            path = 'speed/model_{}_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(args.model, args.placement, local_batch_size, args.placement, fronzen_layer_num)
            # if os.path.exists(path): 
            #     continue 
            for i in range(epoch_frequency):
                start_time = getCurTime()
                train(epoch) 
                time_list.append((getCurTime() - start_time))
                training_time += time_list[-1]
                metric_list.append(copy.deepcopy(_metrics_state())) 
                walkover(trainloader)

            
            info_dict['freeze_layer'] = fronzen_layer_num
            info_dict['list_{}'.format(fronzen_layer_num)] = time_list
            info_dict['metric_{}'.format(fronzen_layer_num)] = metric_list 

            if adaptdl.env.replica_rank() == 0:
                print('saving speed info')
                with open('speed/model_{}_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(args.model, args.placement, local_batch_size, args.placement, fronzen_layer_num), 'wb') as f:
                    np.save(f, info_dict) 
