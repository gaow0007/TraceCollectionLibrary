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
from logging import PlaceHolder
from adaptdl.torch.frozen_layer_count import current_fronzen_layer
from adaptdl.torch.layer_info import profile_layer_info 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import os
os.environ['TARGET_BATCH_SIZE'] = "128"
import argparse
import time
import numpy as np
import copy 
import collections

from PIL import Image
from models import *

import adaptdl
import adaptdl.torch as adl
from adaptdl.torch.utils.misc import collect_atomic_layer_num, apply_frozen, cal_frozen_layer, frozen_net

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


import psutil
memory_cost_percent = 1 - psutil.virtual_memory()[4] / psutil.virtual_memory()[0] 


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
parser.add_argument('--cache', dest='cache', default=False,
                    action='store_true', help='autoscale batchsize')
parser.add_argument('--rewrite', dest='rewrite', default=False,
                    action='store_true', help='autoscale batchsize')

parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False,
                    action='store_true', help='autoscale batchsize')
parser.add_argument('--mixed-precision', dest='mixed_precision', default=False,
                    action='store_true', help='use automatic mixed precision')
args = parser.parse_args()
print(args)
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

class CIFAR10withCache(torchvision.datasets.CIFAR10): 
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10withCache, self).__init__(root, train, transform, target_transform, download)
        self.start_forward_func_id = -1 
        self.cache_frozen_layer = -1 
        self._enable_cache = False 
        self.act_cache = None 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target = self.targets[index] 

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self._enable_cache and self.start_forward_func_id > 0: 
            img = self.act_cache.load_cache_feature(self.cache_frozen_layer, [index], None).squeeze(0) 
        else: 
            img = self.data[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

validset = CIFAR10withCache(root='./data', train=False, download=False, transform=transform_test)
validloader = adl.AdaptiveDataLoader(validset, batch_size=100, shuffle=False, num_workers=2)

if adaptdl.env.replica_rank() == 0:
    trainset = CIFAR10withCache(root='./data', train=True, download=False, transform=transform_train)
    trainloader = adl.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=4, drop_last=False)
    dist.barrier()  # We use a barrier here so that non-master replicas would wait for master to download the data
else:
    dist.barrier()
    trainset = CIFAR10withCache(root='./data', train=True, download=False, transform=transform_train)
    trainloader = adl.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=4, drop_last=False)

# if args.autoscale_bsz:
#     trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1024), gradient_accumulation=False)
# trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1024), gradient_accumulation=False)


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
lr_scheduler = MultiStepLR(optimizer, [50, 75], 0.1)

if args.mixed_precision:
    scaler = torch.cuda.amp.GradScaler(enabled=True)
else:
    scaler = None

# profile layer info 
net.eval() 
placeholder = torch.randn(1, 3, 32, 32)
if not args.rewrite and not args.cache: 
    profile_layer_info(net, placeholder=placeholder) 
net = adl.AdaptiveDataParallel(net, optimizer, lr_scheduler, enable_rewrite=args.rewrite, enable_cache=args.cache, dataset_sample_size=50000, placeholder=placeholder, find_unused_parameters=True)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    dataloader_cost = 0 
    net.train()
    frozen_net(net)
    stats = adl.Accumulator()
    if args.cache: 
        net.epoch_cache_update(trainset)

    train_cost = 0 
    for idx, (index_sample_list, inputs, targets) in enumerate(trainloader): 

        start = getCurTime() 
        optimizer.zero_grad()
        net.cache(index_sample_list) 
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
        net.uncache() 

        stats["loss_sum"] += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        stats["total"] += targets.size(0)
        stats["correct"] += predicted.eq(targets).sum().item()
        train_cost += getCurTime() - start 

    if hasattr(net, 'train_cost'): 
        print('net train time cost is {}'.format(net.time_cost)) 
    print('epoch train time cost is {}'.format(train_cost)) 
    
    if False: 
        start = time.time() 
        for idx, (index_sample_list, inputs, targets) in enumerate(trainloader): 
            pass 
        print('data loader time cost is {}'.format(time.time() - start)) 
    if args.cache:
        net.epoch_cache_roolback(trainset)
        # net.update_cache_frozen_layer(current_fronzen_layer()) # TODO: 


    trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data/")
    net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model/")
    if args.mixed_precision:
        writer.add_scalar("MixedPrecision/scale", scaler.get_scale(), epoch)
    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
        print("Train:", stats)


def valid(epoch):
    net.eval()
    stats = adl.Accumulator()
    with torch.no_grad():
        for _, inputs, targets in validloader:
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
        print("Valid:", stats)


print(adaptdl.env.job_id())
tot_layer_num = collect_atomic_layer_num(net)
tensorboard_dir = 'log_dir'

def getCurTime():
    # torch.cuda.synchronize(0)
    return time.time() 

info_dict = {
    'batch_size': args.bs, 
}
profile_tot_layer = int(tot_layer_num * 0.9) // 2
from adaptdl.torch._metrics import _metrics_state

net.start_cache_manager()
with SummaryWriter(tensorboard_dir) as writer:
    for epoch in adl.remaining_epochs_until(args.epochs):
        # fronzen_layer_num = epoch % profile_tot_layer * 2 * 5 # cal_frozen_layer(epoch, args.epochs, tot_layer_num) # epoch % 20
        fronzen_layer_num =  cal_frozen_layer(epoch, args.epochs, tot_layer_num) 
        fronzen_layer_num = min(fronzen_layer_num, profile_tot_layer)
        # if epoch < 2: 
        #     fronzen_layer_num = 0 
        # else: 
        #     fronzen_layer_num = 10 
        if fronzen_layer_num != current_fronzen_layer(): 
            adl.set_current_frozen_layer(fronzen_layer_num) 
            if args.rewrite: 
                net.stage_rewrite_update()

        apply_frozen(net, fronzen_layer_num)
        print('frozen_layer == {}, rank == {}'.format(fronzen_layer_num, adaptdl.env.replica_rank()))
        epoch_frequency = 1 
        training_time = 0
        time_list = list()
        metric_list = list() 
        metric_info = _metrics_state()
        metric_info.profile = collections.defaultdict(collections.Counter)
        for i in range(epoch_frequency):
            start_time = getCurTime()
            train(epoch)
            valid(epoch)
            time_list.append((getCurTime() - start_time))
            training_time += time_list[-1]
            metric_list.append(copy.deepcopy(_metrics_state()))
        
        # print('frozen_layer == {}'.format(fronzen_layer_num))
        print(time_list)
        
        info_dict[fronzen_layer_num] = training_time / epoch_frequency
        info_dict['list_{}'.format(fronzen_layer_num)] = time_list
        info_dict['metric_{}'.format(fronzen_layer_num)] = metric_list 

net.close_cache_manager()
net.stage_rewrite_rollback(0) 

# if adaptdl.env.replica_rank() == 0:
#     with open('frozen_info/fully_frozen_info_arch_{}_bs_{}_gpu_{}.npy'.format(args.model, args.bs, adaptdl.env.num_replicas()), 'wb') as f:
#         np.save(f, info_dict)
