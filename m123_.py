import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import test_1_pre,test_2_pre,test_3_pre,test_4_end,CNN4_b,CNN1_a,CNN3_c,CNN2_d
# from random_erasing import RandomErasing_vertical, RandomErasing_2x2
import math
import copy

def make_model(args):
    return M123_(args)
######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        # init.constant(m.bias.data, 0.0)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)

    

def pcb_block(num_ftrs, num_stripes, local_conv_out_channels, num_classes, avg=False):
    if avg:
        pooling_list = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(num_stripes)])
    else:
        pooling_list = nn.ModuleList([nn.AdaptiveMaxPool2d(1) for _ in range(num_stripes)])
    conv_list = nn.ModuleList([nn.Conv2d(num_ftrs, local_conv_out_channels, 1, bias=False) for _ in range(num_stripes)])
    batchnorm_list = nn.ModuleList([nn.BatchNorm2d(local_conv_out_channels) for _ in range(num_stripes)])
    relu_list = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(num_stripes)])
    fc_list = nn.ModuleList([nn.Linear(local_conv_out_channels, num_classes, bias=False) for _ in range(num_stripes)])
    for m in conv_list:
        weight_init(m)
    for m in batchnorm_list:
        weight_init(m)
    for m in fc_list:
        weight_init(m)
    return pooling_list, conv_list, batchnorm_list, relu_list, fc_list


def spp_vertical(feats, pool_list, conv_list, bn_list, relu_list, fc_list, num_strides, feat_list=[], logits_list=[]):
    for i in range(num_strides):
        pcb_feat = pool_list[i](feats[:, :, i * int(feats.size(2) / num_strides): (i+1) *  int(feats.size(2) / num_strides), :])
        pcb_feat = conv_list[i](pcb_feat)
        pcb_feat = bn_list[i](pcb_feat)
        pcb_feat = relu_list[i](pcb_feat)
        pcb_feat = pcb_feat.view(pcb_feat.size(0), -1)
        feat_list.append(pcb_feat)
        logits_list.append(fc_list[i](pcb_feat))
    return feat_list, logits_list

def global_pcb(feats, pool, conv, bn, relu, fc, feat_list=[], logits_list=[]):
    global_feat = pool(feats)
    global_feat = conv(global_feat)
    global_feat = bn(global_feat)
    global_feat = relu(global_feat)
    global_feat = global_feat.view(feats.size(0), -1)
    print('222222222222222222222')
    feat_list.append(global_feat)
    logits_list.append(fc(global_feat))
    print('11111111111111111111111')
    return feat_list, logits_list

class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __iter__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = in_channels

    def __next__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class M123_(nn.Module):
    def __init__(self, num_classes, num_stripes=4, local_conv_out_channels=32, feats=256 ,erase=0, loss={'softmax'}, avg=False, **kwargs):
        super(M123_, self).__init__()
        self.erase = erase
        self.num_stripes = num_stripes
        self.loss = loss
        local_conv_out_channels = 32
        num_classes = 751
        feats=256

        # model_ft_b = test_4_end(pretrained=True)
        # cnnb = CNN4_b(num_classes=1000,**kwargs)
        # pretrained_dict = model_ft_b.state_dict()
        # model_dict = cnnb.state_dict()
        # pretrained_dict = { k:v for k,v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # cnnb.load_state_dict(model_dict)

        # model_ft_a = test_1_pre(pretrained=True)
        # cnna = CNN1_a(num_classes=1000,**kwargs)
        # # print(cnna)
        # pretrained_dict = model_ft_a.state_dict()
        # model_dict = cnna.state_dict()
        # pretrained_dict = { k:v for k,v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # cnna.load_state_dict(model_dict)
        # print('a 移除操作已完成')
        # print(cnna)

        # model_ft_c = test_3_pre(pretrained=True)
        # cnnc = CNN3_c(num_classes=1000,**kwargs)
        # # print(cnnc)
        # pretrained_dict = model_ft_c.state_dict()
        # model_dict = cnnc.state_dict()
        # pretrained_dict = { k:v for k,v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # cnnc.load_state_dict(model_dict)
        # print('c 移除操作已完成')
        # print(cnnc)





        model_ft_d = test_2_pre(pretrained=True)
        cnnd = CNN2_d(num_classes=1000,**kwargs)
        # print(cnnd)
        pretrained_dict = model_ft_d.state_dict()
        model_dict = cnnd.state_dict()
        pretrained_dict = { k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        cnnd.load_state_dict(model_dict)
        print('d 移除操作已完成')
        print(cnnd)       

        del model_ft_d
        # print('1ceshi')
        self.num_ftrs = list(cnnd.conv4)[-1].conv1.in_channels
        # self.features_a = cnna
        # self.features_b = cnnb
        # self.features_c = cnnc
        self.features_d = cnnd
        # PSP
        # self.psp_pool, self.psp_conv, self.psp_bn, self.psp_relu, self.psp_upsample, self.conv = psp_block(self.num_ftrs)

        # global
        # self.global_pooling = nn.AdaptiveMaxPool2d(1)
        # self.global_conv = nn.Conv2d(self.num_ftrs, local_conv_out_channels, 1, bias=False)
        # self.global_bn = nn.BatchNorm2d(local_conv_out_channels)
        # self.global_relu = nn.ReLU(inplace=True)
        # self.global_fc = nn.Linear(int(local_conv_out_channels), num_classes, bias=False)

        # weight_init(self.global_conv)
        # weight_init(self.global_bn) 
        # weight_init(self.global_fc)


        # # 2x
        # self.pcb2_pool_list, self.pcb2_conv_list, self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list = pcb_block(self.num_ftrs, 2, local_conv_out_channels, num_classes, avg)
        # # 4x
        # self.pcb4_pool_list, self.pcb4_conv_list, self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list = pcb_block(self.num_ftrs, 4, local_conv_out_channels, num_classes, avg)
        # # 8x
        # self.pcb8_pool_list, self.pcb8_conv_list, self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list = pcb_block(self.num_ftrs, 8, local_conv_out_channels, num_classes, avg)


        pool2d = nn.AvgPool2d
        self.avgpool_p1 = pool2d(kernel_size=(24, 8))
        self.avgpool_p2 = pool2d(kernel_size=(12, 8))
        # self.avgpool_p4 = pool2d(kernel_size=(6, 8))
        self.avgpool_p8 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(512, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        # self.reduction_3 = copy.deepcopy(reduction)
        # self.reduction_4 = copy.deepcopy(reduction)
        # self.reduction_5 = copy.deepcopy(reduction)
        # self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)
        self.reduction_8 = copy.deepcopy(reduction)
        self.reduction_9 = copy.deepcopy(reduction)
        # self.reduction_10 = copy.deepcopy(reduction)
        # self.reduction_11 = copy.deepcopy(reduction)
        # self.reduction_12 = copy.deepcopy(reduction)
        # self.reduction_13 = copy.deepcopy(reduction)
        # self.reduction_14 = copy.deepcopy(reduction)

        self.fc_id_256_0 = nn.Linear(256, num_classes)
        self.fc_id_256_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2 = nn.Linear(256, num_classes)
        # self.fc_id_256_3 = nn.Linear(256, num_classes)
        # self.fc_id_256_4 = nn.Linear(256, num_classes)
        # self.fc_id_256_5 = nn.Linear(256, num_classes)
        # self.fc_id_256_6 = nn.Linear(256, num_classes)
        self.fc_id_256_7 = nn.Linear(256, num_classes)
        self.fc_id_256_8 = nn.Linear(256, num_classes)
        self.fc_id_256_9 = nn.Linear(256, num_classes)
        # self.fc_id_256_10 = nn.Linear(256, num_classes)
        # self.fc_id_256_11 = nn.Linear(256, num_classes)
        # self.fc_id_256_12 = nn.Linear(256, num_classes)
        # self.fc_id_256_13 = nn.Linear(256, num_classes)
        # self.fc_id_256_14 = nn.Linear(256, num_classes)

        self._init_fc(self.fc_id_256_0)
        self._init_fc(self.fc_id_256_1)
        self._init_fc(self.fc_id_256_2)
        # self._init_fc(self.fc_id_256_3)
        # self._init_fc(self.fc_id_256_4)
        # self._init_fc(self.fc_id_256_5)
        # self._init_fc(self.fc_id_256_6)
        self._init_fc(self.fc_id_256_7)
        self._init_fc(self.fc_id_256_8)
        self._init_fc(self.fc_id_256_9)
        # self._init_fc(self.fc_id_256_10)
        # self._init_fc(self.fc_id_256_11)
        # self._init_fc(self.fc_id_256_12)        
        # self._init_fc(self.fc_id_256_13)        
        # self._init_fc(self.fc_id_256_14)
        
        # print('^&%^%&(%^&#$%^&*#$%^&$%^&*')

    def _init_reduction(self,reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)       

    def forward(self, x):
        feat_list = []
        logits_list = []
        # feats_a = self.features_a(x) # N, C, H, W d:r3 blockd  c:r5 blockc  b: r7 blockb a:r9 blocka
        # feats_b = self.features_b(x)
        # feats_c = self.features_c(x)
        feats_d = self.features_d(x)
        
        # assert feats_a.size(2) == 24
        # assert feats_a.size(-1) == 8
        # assert feats_a.size(2) % self.num_stripes == 0
        

        p1 = self.avgpool_p1(feats_d)


        p2 = self.avgpool_p2(feats_d)
        z0_p2 = p2[:, :, 0:1, :]
        z1_p2 = p2[:, :, 1:2, :]

        # p4 = self.avgpool_p4(feats_c)
        # z0_p4 = p4[:, :, 0:1, :]
        # z1_p4 = p4[:, :, 1:2, :]
        # z2_p4 = p4[:, :, 2:3, :]
        # z3_p4 = p4[:, :, 3:4, :]

        p8 = self.avgpool_p8(feats_d)
        z0_p8 = p8[:, :, 0:1, :]
        z1_p8 = p8[:, :, 1:2, :]
        z2_p8 = p8[:, :, 2:3, :]
        # z3_p8 = p8[:, :, 3:4, :]
        # z4_p8 = p8[:, :, 4:5, :]
        # z5_p8 = p8[:, :, 5:6, :]
        # z6_p8 = p8[:, :, 6:7, :]
        # z7_p8 = p8[:, :, 7:8, :]

        f1_p1 = self.reduction_0(p1).squeeze(dim=3).squeeze(dim=2)
        f2_p1 = self.reduction_1(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f2_p2 = self.reduction_2(z1_p2).squeeze(dim=3).squeeze(dim=2)
        # f4_p1 = self.reduction_3(z0_p4).squeeze(dim=3).squeeze(dim=2)
        # f4_p2 = self.reduction_4(z1_p4).squeeze(dim=3).squeeze(dim=2)
        # f4_p3 = self.reduction_5(z2_p4).squeeze(dim=3).squeeze(dim=2)
        # f4_p4 = self.reduction_6(z3_p4).squeeze(dim=3).squeeze(dim=2)
        f8_p1 = self.reduction_7(z0_p8).squeeze(dim=3).squeeze(dim=2)
        f8_p2 = self.reduction_8(z1_p8).squeeze(dim=3).squeeze(dim=2)
        f8_p3 = self.reduction_9(z2_p8).squeeze(dim=3).squeeze(dim=2)
        # f8_p4 = self.reduction_10(z3_p8).squeeze(dim=3).squeeze(dim=2)
        # f8_p5 = self.reduction_11(z4_p8).squeeze(dim=3).squeeze(dim=2)
        # f8_p6 = self.reduction_12(z5_p8).squeeze(dim=3).squeeze(dim=2)
        # f8_p7 = self.reduction_13(z6_p8).squeeze(dim=3).squeeze(dim=2)
        # f8_p8 = self.reduction_14(z7_p8).squeeze(dim=3).squeeze(dim=2)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l1_p1 = self.fc_id_256_0(f1_p1)
        l2_p1 = self.fc_id_256_1(f2_p1)
        l2_p2 = self.fc_id_256_2(f2_p2)   
        # l4_p1 = self.fc_id_256_3(f4_p1)
        # l4_p2 = self.fc_id_256_4(f4_p2)
        # l4_p3 = self.fc_id_256_5(f4_p3)
        # l4_p4 = self.fc_id_256_6(f4_p4)
        l8_p1 = self.fc_id_256_7(f8_p1)
        l8_p2 = self.fc_id_256_8(f8_p2)
        l8_p3 = self.fc_id_256_9(f8_p3)
        # l8_p4 = self.fc_id_256_10(f8_p4)
        # l8_p5 = self.fc_id_256_11(f8_p5)
        # l8_p6 = self.fc_id_256_12(f8_p6)
        # l8_p7 = self.fc_id_256_13(f8_p7)
        # l8_p8 = self.fc_id_256_14(f8_p8)

        f1 = [f1_p1, f2_p1,f2_p2,f8_p1, f8_p2, f8_p3 ]
        l1 = [l1_p1, l2_p1,l2_p2,l8_p1, l8_p2, l8_p3 ]
        predict = torch.cat([f1_p1, f2_p1,f2_p2,f8_p1, f8_p2, f8_p3], dim=1)

        return predict, f1,l1



        # if self.erase>0:
        #    print('Random Erasing')
            # erasing = RandomErasing_vertical(probability=self.erase)
            # feats_a = erasing(feats_a)
            # feats_b = erasing(feats_b)
            # feats_c = erasing(feats_c)
            # feats_d = erasing(feats_d)
        
        # feat_list, logits_list = global_pcb(feats_a, self.global_pooling, self.global_conv, self.global_bn, 
        #             self.global_relu, self.global_fc, [], [])
        # feat_list, logits_list = spp_vertical(feats_b, self.pcb2_pool_list, self.pcb2_conv_list, 
        #             self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list, 2, feat_list, logits_list)
        # feat_list, logits_list = spp_vertical(feats_c, self.pcb4_pool_list, self.pcb4_conv_list, 
        #             self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list, 4, feat_list, logits_list)

        # feat_list, logits_list = spp_vertical(feats_d, self.pcb8_pool_list, self.pcb8_conv_list, 
        #             self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list, 8, feat_list, logits_list)

        # print('SHUZU已得到')
        # if not self.training:
        #     return torch.cat(feat_list, dim=1)
        
        # predict = torch.cat(feat_list, dim=1)

        # return predict,feat_list,logits_list




        # if self.loss == {'softmax'}:
        #     return logits_list
        # elif self.loss == {'xent'}:
        #     return logits_list
        # elif self.loss == {'xent', 'htri'}:
        #     return logits_list, feat_list
        # elif self.loss == {'cent'}:
        #     return logits_list, feat_list
        # elif self.loss == {'ring'}:
        #     return logits_list, feat_list
        # else:
        #     raise KeyError("Unsupported loss: {}".format(self.loss))