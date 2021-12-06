import logging
import utils.gpu as gpu
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
import config.yolov3_config_voc as cfg
from utils import cosine_lr_scheduler

import adaptdl
import adaptdl.env
import adaptdl.torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from adaptdl.torch.layer_info import profile_layer_info 
from torch.utils.tensorboard import SummaryWriter

# from apex import amp
# from apex.amp._amp_state import _amp_state

from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics, get_progress

# added for frozen 
from adaptdl.torch.utils.misc import collect_atomic_layer_num, apply_frozen, cal_frozen_layer, frozen_net
from adaptdl.torch.frozen_layer_count import current_fronzen_layer
import numpy as np
import random
import copy 

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)



def set_local_batch_size(local_batch_size): 
    batch_size = local_batch_size * adaptdl.env.num_replicas() 
    os.environ['TARGET_BATCH_SIZE'] = "{}".format(batch_size) 



def walkover(trainloader): 
    for idx, _ in enumerate(trainloader):  
        pass



class Trainer(object):
    def __init__(self, weight_path, placement):
        init_seeds(0)
        self.placement = placement 
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = False # cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = adaptdl.torch.AdaptiveDataLoader(self.train_dataset,
                                                                 batch_size=cfg.TRAIN["BATCH_SIZE"],
                                                                 num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                                                 drop_last=True,
                                                                 shuffle=True)
        # self.train_dataloader.autoscale_batch_size(512, local_bsz_bounds=(4, 8),
        #                                            gradient_accumulation=True)
        self.valid_dataset = data.VocDataset(anno_file_type="test")
        self.valid_dataloader = adaptdl.torch.AdaptiveDataLoader(self.valid_dataset,
                                                                 batch_size=(8 * adaptdl.env.num_replicas()),
                                                                 num_workers=8,
                                                                 shuffle=False)
        self.yolov3 = Yolov3().cuda()

        self.optimizer = optim.SGD(self.yolov3.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])

        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.yolov3.load_darknet_weights(weight_path)

        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max=(self.epochs - cfg.TRAIN["WARMUP_EPOCHS"]),
                                           eta_min=cfg.TRAIN["LR_END"])
        adaptdl.torch.init_process_group("nccl")
        profile_layer_info(self.yolov3, placeholder=torch.randn(1, 3, 416, 416)) 
        # self.yolov3, self.optimizer = amp.initialize(self.yolov3, self.optimizer) 
        # 
        # self.yolov3 = adaptdl.torch.AdaptiveDataParallel(self.yolov3, self.optimizer, self.scheduler,
        #                                                  patch_optimizer=False)
        self.yolov3 = adaptdl.torch.AdaptiveDataParallel(self.yolov3, self.optimizer, lr_scheduler, enable_rewrite=opt.rewrite, \
            enable_cache=opt.cache, dataset_sample_size=-1, placeholder=None, find_unused_parameters=True)


        self.yolov3.adascale._smoothing = 0.997


    def epoch_train(self, limit=50): 
        self.yolov3.train()
        frozen_net(self.yolov3)
        accum = adaptdl.torch.Accumulator()
        for idx, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(self.train_dataloader):
            if idx > limit: 
                break  
            imgs = imgs.cuda()
            label_sbbox = label_sbbox.cuda()
            label_mbbox = label_mbbox.cuda()
            label_lbbox = label_lbbox.cuda()
            sbboxes = sbboxes.cuda()
            mbboxes = mbboxes.cuda()
            lbboxes = lbboxes.cuda()

            p, p_d = self.yolov3(imgs)

            loss, loss_giou, loss_conf, loss_cls = self.criterion(
                    p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)

            delay_unscale = not self.train_dataloader._elastic.is_sync_step() 
            loss.backward() 
            self.yolov3.adascale.step()

            accum["loss_sum"] += loss.item() * imgs.size(0)
            accum["loss_cnt"] += imgs.size(0) 


            # Multi-scale training (320-608 pixels).
            if self.multi_scale_train:
                self.train_dataset.img_size = random.choice(range(10, 20)) * 32 
            del imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
            del p, p_d, loss, loss_giou, loss_conf, loss_cls


    def train(self): 
        print("Train datasets number is : {}".format(len(self.train_dataset))) 
        # added by frozen 
        self.tot_layer_num = collect_atomic_layer_num(self.yolov3) 
        self.profile_tot_layer = int(self.tot_layer_num * 0.7) 
        from adaptdl.torch._metrics import _metrics_state
        self.yolov3.start_cache_manager()
        
        for local_batch_size in [4, 6, 8, 12, 16]: 
            # set_local_batch_size(local_batch_size=local_batch_size)
            frozen_list = np.linspace(0, self.profile_tot_layer, 10)
            frozen_list = [int(layer) for layer in frozen_list] 
            for epoch, frozen_layer_num in enumerate(frozen_list): 
                path = 'speed/model_yolo_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(self.placement, local_batch_size, self.placement, frozen_layer_num)
                if os.path.exists(path): 
                    continue 
                from adaptdl.torch.data import AdaptiveDataLoaderHelper
                AdaptiveDataLoaderHelper._training = None 
                self.train_dataloader = adaptdl.torch.AdaptiveDataLoader(self.train_dataset,
                                                                        batch_size=cfg.TRAIN["BATCH_SIZE"],
                                                                        num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                                                        drop_last=False,
                                                                        shuffle=True) 
                self.train_dataloader._elastic.train() 
                set_local_batch_size(local_batch_size=local_batch_size)

                adaptdl.torch.set_current_frozen_layer(frozen_layer_num)
                apply_frozen(self.yolov3, frozen_layer_num)
                metric_list = list() 
                metric_info = _metrics_state()
                info_dict = dict() 
                metric_info.profile = collections.defaultdict(collections.Counter) 

                self.epoch_train()
                metric_list.append(copy.deepcopy(_metrics_state())) 
                # walkover(self.train_dataloader)
                info_dict['freeze_layer'] = frozen_layer_num 
                info_dict['metric_{}'.format(frozen_layer_num)] = metric_list

                if adaptdl.env.replica_rank() == 0:
                    print('saving speed info')
                    with open('speed/model_yolo_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(self.placement, local_batch_size, self.placement, frozen_layer_num), 'wb') as f:
                        np.save(f, info_dict) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', dest='cache', default=False,
                        action='store_true', help='cache operation')
    parser.add_argument('--rewrite', dest='rewrite', default=False,
                        action='store_true', help='rewrite operation')
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    parser.add_argument('--placement', type=str, default=None, help='placement position')
    opt = parser.parse_args()

    Trainer(weight_path=opt.weight_path, placement=opt.placement).train()
