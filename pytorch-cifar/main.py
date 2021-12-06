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
# os.environ['TARGET_BATCH_SIZE'] = "128"
os.environ['TRACE_EFFICIENCY'] = "True"
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
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False,
                    action='store_true', help='autoscale batchsize')
parser.add_argument('--frozen', dest='frozen', default=False,
                    action='store_true', help='autoscale batchsize')
parser.add_argument('--mixed-precision', dest='mixed_precision', default=False,
                    action='store_true', help='use automatic mixed precision')
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


# if args.mixed_precision:
#     scaler = torch.cuda.amp.GradScaler(enabled=True)
# else:
#     scaler = None

# profile layer info 
adl.profile_layer_info(net, torch.randn(1, 3, 32, 32))
net = adl.AdaptiveDataParallel(net, optimizer, lr_scheduler, find_unused_parameters=True) 
adaptdl.torch.current_fronzen_layer() 


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    frozen_net(net)
    stats = adl.Accumulator()
    for idx, (inputs, targets) in enumerate(trainloader):  
        optimizer.zero_grad() 
        if args.mixed_precision:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    if args.mixed_precision:
        writer.add_scalar("MixedPrecision/scale", scaler.get_scale(), epoch)
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
    # import pdb; pdb.set_trace() 
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
            

print(adaptdl.env.job_id())
tot_layer_num = collect_atomic_layer_num(net)
tensorboard_dir = 'log_dir'

stats = {
    'epoch': list(), 
    'metric': list(), 
    'progress': list(), 
    'grad_var': list(), 
    'grad_sqr': list(), 
    'iteration': list(), 
}


start_idx = 0
layer_num_idx = 0
module_idx_len = dict()
for name, module in net.named_modules():
    if isAtomicLayer(module):
        module_idx_len[name] = len([param for param in module.parameters()])
        stats['layer_{}_grad_var'.format(layer_num_idx)] = list()
        stats['layer_{}_grad_sqr'.format(layer_num_idx)] = list() 
        layer_num_idx += 1

with SummaryWriter(tensorboard_dir) as writer:
    for epoch in adl.remaining_epochs_until(args.epochs):
        if not args.frozen:
            fronzen_layer_num = 0
        else:
            fronzen_layer_num = cal_frozen_layer(epoch, args.epochs, tot_layer_num) 
        
        adl.set_current_frozen_layer(fronzen_layer_num)
        apply_frozen(net, fronzen_layer_num)

        start_time = time.time()
        iteration = train(epoch)
        acc = valid(epoch)

        lr_scheduler.step()
        if adaptdl.env.replica_rank() == 0:
            stats['epoch'].append(epoch)
            stats['metric'].append(acc)
            stats['progress'].append(get_progress())
            sqr_avg = net.adascale._optimizer.state["adascale"]["sqr_avg"]
            var_avg = net.adascale._optimizer.state["adascale"]["var_avg"]
            stats['grad_var'] = sum(var_avg)
            stats['grad_sqr'] = sum(sqr_avg)
            print('stats grad_var {}'.format(stats['grad_var']))
            print('stats grad_sqr {}'.format(stats['grad_sqr']))
            print('progress {}'.format(stats['progress'][-1]))
            if len(stats['iteration']) == 0:
                stats['iteration'] = [iteration]
            else: 
                stats['iteration'].append(stats['iteration'][-1] + iteration)
            
            layer_num_idx = 0
            start_idx = 0
            for name, module in net.named_modules():
                if isAtomicLayer(module):
                    sub_var = 0
                    sub_sqr = 0
                    for j in range(start_idx, start_idx + module_idx_len[name]):
                        sub_var += var_avg[j]
                        sub_sqr += sqr_avg[j]
                    stats['layer_{}_grad_var'.format(layer_num_idx)].append(sub_var)
                    stats['layer_{}_grad_sqr'.format(layer_num_idx)].append(sub_sqr)
                    layer_num_idx += 1 
                    start_idx += module_idx_len[name]
        


if adaptdl.env.replica_rank() == 0:
    filename = 'stats/model_{}_bs_{}_frozen_{}'.format(args.model, args.bs * adaptdl.env.num_replicas(), args.frozen)
    with open(filename, 'wb') as f:
        np.save(f, stats)
