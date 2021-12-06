import sys
sys.path.append("..")

# AbsolutePath = os.path.abspath(__file__)           #将相对路径转换成绝对路径
# SuperiorCatalogue = os.path.dirname(AbsolutePath)   #相对路径的上级路径
# BaseDir = os.path.dirname(SuperiorCatalogue)        #在“SuperiorCatalogue”的基础上在脱掉一层路径，得到我们想要的路径。
# sys.path.insert(0,BaseDir)                          #将我们取出来的路径加入

import torch.nn as nn
import torch
from model.backbones.darknet53 import Darknet53
from model.necks.yolo_fpn import FPN_YOLOV3, CONV_YOLOV3
from model.head.yolo_head import Yolo_head
from model.layers.conv_module import Convolutional
import config.yolov3_config_voc as cfg
import numpy as np
from utils.tools import *


class Yolov3(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, init_weights=True):
        super(Yolov3, self).__init__()

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        self.__backnone = Darknet53() 
        self.__fpn = CONV_YOLOV3(fileters_in=[1024, 512, 256],
                                fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel]) 
        # self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
        #                         fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        # # small
        # self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # # medium
        # self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        if init_weights:
            self.__init_weights()


    def forward(self, x):
        # out = []
        x_l = self.__backnone(x) 
        x_l = self.__fpn(x_l) 
        p, p_d = self.__head_l(x_l)
        out = p, p_d 
        return out 

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                # print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                # print("initing {}".format(m))


    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        print("load darknet weights : ", weight_file) 
        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    # print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                # print("loading weight {}".format(conv_layer))



if __name__ == '__main__':
    net = Yolov3()
    
    # print(net)
    net.eval() 
    in_img = torch.randn(12, 3, 416, 416)
    p, p_d = net(in_img) 
    # exit(0)

    net.eval() 
    from adaptdl.torch.rewriter.graph_rewriter import rewrite_graph, split_graph 
    from adaptdl.torch.rewriter.rewriter import fuse 
    import adaptdl
    model = net# ._Yolov3__backnone
    # gm = adaptdl.torch.fx.symbolic_trace(model) 
    # print(gm)
    # exit(0) 
    if False: 
        import copy 
        gm = adaptdl.torch.fx.symbolic_trace(model) 
        model, forward_func_length, critical_operation_layer = split_graph(model) 
        meta_model = copy.deepcopy(model) 
        fused_model, fuse_pairs = fuse(meta_model, fake=True) 

        ori_forward_func = meta_model.__class__.forward 
        idx = 0
        setattr(meta_model.__class__, 'forward', getattr(meta_model.__class__, 'forward_{}'.format(idx))) 
        meta_gm = adaptdl.torch.fx.symbolic_trace(meta_model) 
        exit(0)

    model, forward_func_length, critical_operation_layer = rewrite_graph(model, zero_copy=True) 
    for i in range(forward_func_length): 
        func = getattr(model, 'forward_{}'.format(i))
        import inspect 
        lines = inspect.getsource(func)
        print(lines)
        print()

    exit(0)
    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)
