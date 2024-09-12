# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from collections import OrderedDict
import copy
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class cosLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(cosLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        self.scale = 0.09



    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.000001)

        L_norm = torch.norm(self.L.weight, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        weight_normalized = self.L.weight.div(L_norm + 0.000001)
        cos_dist = torch.mm(x_normalized,weight_normalized.transpose(0,1))
        scores = cos_dist / self.scale
        return scores



def resnet18_reduced(nclasses: int, nf: int = 20,args= None):
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNetGPM(BasicBlockGPM, [2, 2, 2, 2], nclasses, nf=nf,args=args)

def init_weights(model, std=0.01):
    print("Initialize weights of %s with normal dist: mean=0, std=%0.2f" % (type(model), std))
    for m in model.modules():
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 0.1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                m.bias.data.zero_()



class BasicBlockGPM(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockGPM, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNetGPM(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf,taskcla=None,args=None):
        super(ResNetGPM, self).__init__()
        self.args = args
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1,1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        #self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=False)
        expand = 4 
        if self.args.proj_gpm :

            self.simclr = nn.Linear(nf * 8 * block.expansion*expand, 640,bias=False)
            
        else :
            self.simclr = nn.Linear(nf * 8 * block.expansion*expand, 128)

        
        self.task_num = self.args.task_num
        assert num_classes%self.task_num == 0 , 'num_classes%task num != 0'
        self.linear=torch.nn.ModuleList()
        for t in range(self.task_num):
            self.linear.append(cosLinear(nf * 8 * block.expansion *expand , num_classes//self.task_num))
    #self.classifier = self.linear
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        bsz = x.size(0)

        self.act['conv_in'] = x
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.args.dataset=='miniimagenet' :
            out = avg_pool2d(out, 4)
        else :
            out = avg_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        return out

    def logit(self, x):
        '''Apply the last FC linear mapping to get logits'''
        

        y=[]
        for t in range(self.task_num):
            y.append(self.linear[t](x))
        
        
        return y

    def forward(self, x: torch.Tensor, use_proj=False):
        out = self.features(x)
        if use_proj:
            feature = out
            out = self.simclr(out)
            return feature, out
        else:
            out = self.logit(out)
        return out

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def simclr_proj(self,feature):
        
        feature_normalied = feature
        self.act['proj']=feature_normalied
        return self.simclr(feature_normalied)

        
