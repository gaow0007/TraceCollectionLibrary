import logging
import utils.gpu as gpu
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import os
os.environ['TRACE_EFFICIENCY'] = "True"

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
from adaptdl.torch.utils.misc import collect_atomic_layer_num, apply_frozen, cal_frozen_layer, frozen_net, isAtomicLayer
from adaptdl.torch.frozen_layer_count import current_fronzen_layer
import numpy as np
import random




torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class Trainer(object):
    def __init__(self, weight_path):
        init_seeds(0)
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"]) 
        self.train_dataloader = adaptdl.torch.AdaptiveDataLoader(self.train_dataset,
                                                                 batch_size=cfg.TRAIN["BATCH_SIZE"],
                                                                 num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                                                 drop_last=False,
                                                                 shuffle=True)
        # self.train_dataloader.autoscale_batch_size(512, local_bsz_bounds=(4, 8),
        #                                            gradient_accumulation=True)
        self.valid_dataset = data.VocDataset(anno_file_type="test")
        self.valid_dataloader = adaptdl.torch.AdaptiveDataLoader(self.valid_dataset,
                                                                 batch_size=(8 * adaptdl.env.num_replicas()),
                                                                 num_workers=8,
                                                                 shuffle=False)
        self.yolov3 = Yolov3().cuda()

        self.optimizer = optim.SGD([{"params": [param]} for param in self.yolov3.parameters()], lr=cfg.TRAIN["LR_INIT"],
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
        self.init_stats()

    def init_stats(self, ): 
        self.stats = {
            'epoch': list(), 
            'metric': list(), 
            'progress': list(), 
            'grad_var': list(), 
            'grad_sqr': list(), 
            'iteration': list(), 
        }
        layer_num_idx = 0 
        self.module_idx_len = dict()
        for name, module in self.yolov3.named_modules():
            if isAtomicLayer(module):
                self.module_idx_len[name] = len([param for param in module.parameters()])
                self.stats['layer_{}_grad_var'.format(layer_num_idx)] = list()
                self.stats['layer_{}_grad_sqr'.format(layer_num_idx)] = list() 
                layer_num_idx += 1

    def valid(self, epoch):
        self.yolov3.train()

        accum = adaptdl.torch.Accumulator()
        with torch.no_grad():
            for imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes in self.valid_dataloader: 
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

                accum["loss_sum"] += loss.item() * imgs.size(0)
                accum["loss_cnt"] += imgs.size(0)

                # Print batch results
                # print("Epoch {} valid [{}/{}]:  loss_giou: {:.4f}  loss_conf: {:.4f}  loss_cls: {:.4f}  loss: {:.4f}"
                #       .format(epoch, self.valid_dataloader._elastic.current_index, len(self.valid_dataset),
                #               loss_giou.item(), loss_conf.item(), loss_cls.item(), loss.item()))

                del imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
                del p, p_d, loss, loss_giou, loss_conf, loss_cls

        with accum.synchronized(), SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
            accum["loss_avg"] = accum["loss_sum"] / accum["loss_cnt"]
            writer.add_scalar("Loss/Valid", accum["loss_avg"], epoch)
            # report_valid_metrics(epoch, accum["loss_avg"])
            print("Valid:", accum)
            return accum["loss_avg"]

    def epoch_start_frozen(self, epoch): 
        fronzen_layer_num =  cal_frozen_layer(epoch, self.epochs * 2, self.tot_layer_num // 2) 
        fronzen_layer_num = min(fronzen_layer_num, self.profile_tot_layer)  
        fronzen_layer_num = 0
        print('having frozen {}'.format(fronzen_layer_num))
        if fronzen_layer_num != current_fronzen_layer(): 
            adaptdl.torch.set_current_frozen_layer(fronzen_layer_num) 
            if opt.rewrite: 
                self.yolov3.stage_rewrite_update() 
        apply_frozen(self.yolov3, fronzen_layer_num)
        frozen_net(self.yolov3)
        if opt.cache: 
            self.yolov3.epoch_cache_update(self.train_dataset)

    def epoch_collect_stats(self, epoch, iteration, metric): 
        stats = self.stats 
        if adaptdl.env.replica_rank() == 0:
            stats['epoch'].append(epoch)
            stats['metric'].append(metric)
            stats['progress'].append(get_progress())
            sqr_avg = self.yolov3.adascale._optimizer.state["adascale"]["sqr_avg"]
            var_avg = self.yolov3.adascale._optimizer.state["adascale"]["var_avg"]
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
            for name, module in self.yolov3.named_modules():
                if isAtomicLayer(module):
                    sub_var = 0
                    sub_sqr = 0 
                    for j in range(start_idx, start_idx + self.module_idx_len[name]):
                        sub_var += var_avg[j]
                        sub_sqr += sqr_avg[j] 
                    stats['layer_{}_grad_var'.format(layer_num_idx)].append(sub_var)
                    stats['layer_{}_grad_sqr'.format(layer_num_idx)].append(sub_sqr)
                    layer_num_idx += 1 
                    start_idx += self.module_idx_len[name]
        


    
    def epoch_finish_frozen(self, epoch): 
        if opt.cache:
            self.yolov3.epoch_cache_roolback(self.train_dataset)

    def train(self):
        # print(self.yolov3)
        # print("Train datasets number is : {}".format(len(self.train_dataset))) 
        # added by frozen 
        self.tot_layer_num = collect_atomic_layer_num(self.yolov3) 
        self.profile_tot_layer = int(self.tot_layer_num * 0.9) 
        from adaptdl.torch._metrics import _metrics_state
        self.yolov3.start_cache_manager()

        for epoch in adaptdl.torch.remaining_epochs_until(self.epochs): 
            self.epoch_start_frozen(epoch) 
            self.yolov3.train()
            accum = adaptdl.torch.Accumulator()
            with SummaryWriter('log_dir') as writer:
                for idx, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(self.train_dataloader): 
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
                    # with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
                    #     self.yolov3.adascale.loss_scale = _amp_state.loss_scalers[0].loss_scale()
                    #     scaled_loss.backward()
                    loss.backward() 
                    self.yolov3.adascale.step()

                    accum["loss_sum"] += loss.item() * imgs.size(0)
                    accum["loss_cnt"] += imgs.size(0) 

                    # Print batch results
                    if idx % 200 == 0 and adaptdl.env.replica_rank() == 0: 
                        print("Epoch {} train [{}/{}]:  img_size: {}  loss_giou: {:.4f}  loss_conf: {:.4f}  loss_cls: {:.4f}  loss: {:.4f}"
                            .format(epoch, self.train_dataloader._elastic.current_index, len(self.train_dataset),
                                    self.train_dataset.img_size, loss_giou.item(), loss_conf.item(), loss_cls.item(), loss.item()))

                    # Multi-scale training (320-608 pixels).
                    if self.multi_scale_train:
                        self.train_dataset.img_size = random.choice(range(10, 20)) * 32

                    global_step = int(self.yolov3.adascale._state["progress"])
                    self.train_dataloader.to_tensorboard(writer, global_step, tag_prefix="AdaptDL/Data")
                    self.yolov3.to_tensorboard(writer, global_step, tag_prefix="AdaptDL/Model")

                    del imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
                    del p, p_d, loss, loss_giou, loss_conf, loss_cls

                    if epoch < cfg.TRAIN["WARMUP_EPOCHS"]:
                        for group in self.optimizer.param_groups:
                            group["lr"] = (get_progress() / len(self.train_dataset) *
                                           self.train_dataloader.batch_size /
                                           cfg.TRAIN["WARMUP_EPOCHS"]) * cfg.TRAIN["LR_INIT"]
                    # print("lr =", self.optimizer.param_groups[0]["lr"]) 
                iteration = idx + 1
                with accum.synchronized():
                    accum["loss_avg"] = accum["loss_sum"] / accum["loss_cnt"]
                    writer.add_scalar("Loss/Train", accum["loss_avg"], epoch)
                    # report_train_metrics(epoch, accum["loss_avg"])
                    print("Train:", accum)

            metric = self.valid(epoch)

            if epoch >= cfg.TRAIN["WARMUP_EPOCHS"]:
                self.scheduler.step()
            self.epoch_finish_frozen(epoch) 
            self.epoch_collect_stats(epoch, iteration=iteration, metric=metric)

            # if (epoch + 1) % 5 == 0: 
            #     with torch.no_grad(), SummaryWriter('log_dir') as writer:
            #         print('*'*20+"Evaluate"+'*'*20)
            #         APs = Evaluator(self.yolov3).APs_voc()
            #         mAP = 0
            #         for i in APs:
            #             print("{} --> mAP : {}".format(i, APs[i]))
            #             mAP += APs[i]
            #         mAP = mAP / self.train_dataset.num_classes
            #         print('mAP:%g'%(mAP))
            #         writer.add_scalar("Eval/mAP", float(mAP))
        if adaptdl.env.replica_rank() == 0: 
            torch.save(self.yolov3.state_dict(), 'weight/best.pth')
            filename = 'stats/model_yolo_bs_{}'.format(cfg.TRAIN['BATCH_SIZE'] * adaptdl.env.num_replicas())
            with open(filename, 'wb') as f:
                np.save(f, self.stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', dest='cache', default=False,
                        action='store_true', help='cache operation')
    parser.add_argument('--rewrite', dest='rewrite', default=False,
                        action='store_true', help='rewrite operation')
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    parser.add_argument('--batch_size', type=int, default=0, help='target batch size scale') 
    parser.add_argument('--epochs', type=int, default=0, help='target batch size scale') 
    opt = parser.parse_args()
    cfg.TRAIN["EPOCHS"] = opt.epochs
    Trainer(weight_path=opt.weight_path).train()
