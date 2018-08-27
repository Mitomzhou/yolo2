'''
Created on Aug 5, 2018
Reference from marvis, but this version does not use cuda
look through the website if you want to know more
https://github.com/marvis/pytorch-yolo2
'''

import torch
import torch.nn as nn
import numpy as np
import math
import time
import torch.nn.functional as F
from numpy import dtype

from utils import *


class DarkNet(nn.Module):
    """
    
    """
    def __init__(self, cfgfile):
        super(DarkNet, self).__init__()
        self.blocks = parse_config(cfgfile)
        self.models = self.create_network()
        self.loss = self.models[len(self.models)-1]
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])
        self.anchors = self.loss.anchors
        self.num_anchor = self.loss.num_anchor
        self.anchor_step = self.loss.anchor_step
        self.num_class = self.loss.num_class
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        
    def forward(self, X):
        index = -2 # for loop, index = 0 conv first
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            index += 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg':
                X = self.models[index](X)
                outputs[index] = X
                '''
                # save cache to file for evaluate result
                if block['type'] == 'maxpool':
                    self.index += 1
                    if self.index == 4:
                        assert(X.size(0)*X.size(1)*X.size(2)*X.size(3) == 26*26*256)
                        temp_X = X.data
                        temp_X = temp_X.view(1*4, 256, 13*13).transpose(0,1).contiguous().view(256*4, 13*13).transpose(0,1).contiguous()
                        temp_X = temp_X.numpy().astype(np.float64)
                        save_data(temp_X, 'maxpool_11_output.ods')
                '''
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                # layers suport -1 and 1
                layers = [int(i) if int(i) > 0 else index + int(i) for i in layers]
                if len(layers) == 1:
                    X = outputs[layers[0]]
                    outputs[index] = X
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    X = torch.cat((x1, x2), 1) # 0:vertical splicing  1:transverse splicing
                    outputs[index] = X
            elif block['type'] == 'region':
                continue
        return X
        
    def create_network(self):
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = [] 
        conv_id = 0
        for block in self.blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id += 1
                #batch_normalize
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)/2 if is_pad else 0
                activation_fun = block['activation']
                model = nn.Sequential()
                # be careful the last convolutional layer, no batch_normalize and activation_fun
                if batch_normalize:
                    model.add_module('conv', nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('batch_normalize', nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv', nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation_fun == 'leaky':
                    model.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))
                elif activation_fun == 'relu':
                    model.add_module('relu', nn.ReLU())
                prev_filters = filters
                out_filters.append(prev_filters) 
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                model = nn.MaxPool2d(pool_size, stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'reorg':
                # data reconstitution, large resolution to small resolution
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                # combined output 
                layers = block['layers'].split(',')
                model_index = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ model_index for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == model_index -1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(Route())
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_anchor = int(block['num'])
                loss.anchor_step = len(loss.anchors) / loss.num_anchor
                loss.num_class = int(block['classes'])
                loss.object_scale = float(block['object_scale']) # exist object, loss weight parm:5
                loss.noobject_scale = float(block['noobject_scale']) # not exist object, loss weight parm:1
                loss.class_scale = float(block['class_scale']) # 1
                loss.coord_scale = float(block['coord_scale']) # 1
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))
        return models
                
    def print_network(self):
        """
        print network structure
        """
        print('----------------------------darknet------------------------------------')
        print('|      layer  filters  size           input                output     |')
        print('-----------------------------------------------------------------------')
        prev_width = int(self.blocks[0]['width'])
        prev_height = int(self.blocks[0]['height'])
        prev_filters = int(self.blocks[0]['channels'])
        out_widths = [] # for route
        out_heights = [] # for route
        out_filters = [] # for route
        index = -2
        for block in self.blocks:
            index += 1
            if block['type'] == 'net':
                continue
            if block['type'] == 'convolutional':
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)/2 if is_pad else 0
                stride = int(block['stride'])
                width = (prev_width + 2*pad - kernel_size)/stride + 1 # or prev_width
                height = (prev_height + 2*pad - kernel_size)/stride + 1 # or prev_height
                print('%5d %-s    %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                       index, 'conv', filters,  kernel_size, kernel_size, stride,                                                                    
                       prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            if block['type'] == 'maxpool':
                kernel_size = int(block['size']) 
                stride = int(block['stride'])
                width = (prev_width - kernel_size)/stride + 1
                height = (prev_height - kernel_size)/stride + 1
                filters = prev_filters
                print('%5d %-s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                       index, 'maxpool', kernel_size, kernel_size, stride,                                                                    
                       prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            if block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ index for i in layers]
                if len(layers) == 1:
                    prev_width = out_widths[layers[0]]
                    prev_height = out_heights[layers[0]]
                    prev_filters = out_filters[layers[0]]
                    print('%5d %-s     %3d' % (index, 'route', layers[0])) 
                elif len(layers) == 2:
                    prev_width = out_widths[layers[0]]
                    prev_height = out_heights[layers[0]]
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    print('%5d %-s     %3d,%3d' % (index, 'route', layers[0], layers[1])) 
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            if block['type'] == 'reorg':
                stride = int(block['stride'])
                width = prev_width / stride
                height = prev_height / stride
                filters = prev_filters * stride * stride
                print('%5d %-s               / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                       index, 'reorg', stride,                                                                    
                       prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
        print('-----------------------------------------------------------------------')   
                
    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32) 
        fp.close()
        start = 0
        index = -2
        for block in self.blocks:
            if start >= buf.size: # 50983561
                break
            index += 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[index]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_batchnormal(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'region':
                pass
            else:
                print('unknown type %s' % (block['type']))    
            
                
class Reorg(nn.Module):
    """
    - resize data
    """
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, X):
        stride = self.stride
        assert(X.data.dim() == 4)
        N = X.data.size(0)
        C = X.data.size(1)
        W = X.data.size(2)
        H = X.data.size(3)
        assert(W % stride == 0)
        assert(H % stride == 0)
        X = X.view(N, C, H/stride, stride, W/stride, stride).transpose(3,4).contiguous()
        X = X.view(N, C, H/stride*W/stride, stride*stride).transpose(2,3).contiguous()
        X = X.view(N, C, stride*stride, H/stride, W/stride).transpose(1,2).contiguous()
        X = X.view(N, stride*stride*C, H/stride, W/stride)
        return X
        
        
class Route(nn.Module):
    """
    - combined output
    """
    def __init__(self):
        super(Route, self).__init__()
    def forward(self, X):
        return X
 
 
class RegionLoss(nn.Module):
    """
    - for training
    """
    def __init__(self, num_classes=20, anchors=[1.3221, 1.73145, 3.19275, \
            4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071], num_anchors=5):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        """
        - for computing loss between output and target
        """
        #output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = F.sigmoid(output.index_select(2, torch.autograd.Variable(torch.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, torch.autograd.Variable(torch.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, torch.autograd.Variable(torch.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, torch.autograd.Variable(torch.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, torch.autograd.Variable(torch.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, torch.autograd.Variable(torch.linspace(5,5+nC-1,nC).long()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW)
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW)
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data[0])

        tx    = torch.autograd.Variable(tx)
        ty    = torch.autograd.Variable(ty)
        tw    = torch.autograd.Variable(tw)
        th    = torch.autograd.Variable(th)
        tconf = torch.autograd.Variable(tconf)
        tcls  = torch.autograd.Variable(tcls.view(-1)[cls_mask].long())

        coord_mask = torch.autograd.Variable(coord_mask)
        conf_mask  = torch.autograd.Variable(conf_mask.sqrt())
        cls_mask   = torch.autograd.Variable(cls_mask.view(-1, 1).repeat(1,nC))
        cls        = cls[cls_mask].view(-1, nC)  

        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if True:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        return loss
    
    
if __name__ == '__main__':
    net = DarkNet('cfg/yolo.cfg')
    net.load_weights('data/yolo.weights')


