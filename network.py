import json

import cv2
import math
import torch
import torchvision

import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import models
from torchvision.transforms import transforms


#from lib.nms.pth_nms import pth_nms

import torch.nn as nn
import torch
import collections
from abc import ABCMeta, abstractmethod

from anchors import Anchors
from network_base import NetworkBase



class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]

        if deltas.size(2) == 3:
            dh = deltas[:, :, 2] * self.std[3] + self.mean[3]
        else:
            dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes
        
        
        
class ClipBoxes(nn.Module):
    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)


        return boxes


#def nms(dets, thresh):
#    "Dispatch to either CPU or GPU NMS implementations.\
#    Accept dets as tensor"""
#    return pth_nms(dets, thresh)

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


IMG_SIZE = (480, 640)
HEATMAP_SIZE = (29, 39)
TEMPLATE_SIZE = 124

PREPROCESS = [transforms.Compose([transforms.Scale(IMG_SIZE[0]), transforms.ToTensor(), normalize]),
              transforms.Compose([transforms.Scale(TEMPLATE_SIZE), transforms.ToTensor(), normalize]),
              transforms.Compose([transforms.Scale(TEMPLATE_SIZE), transforms.ToTensor()]),
              transforms.Compose([transforms.Scale(TEMPLATE_SIZE), transforms.ToTensor(), normalize]),
              transforms.Compose([transforms.Scale(TEMPLATE_SIZE), transforms.ToTensor()])
              ]

eps = 0.00001



class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, num_classes=2, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ELU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ELU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ELU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes), out


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ELU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ELU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ELU()
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)

#Backbone
class ImageFeatExtract(nn.Module):
    def __init__(self):
        super(ImageFeatExtract, self).__init__()

        dense = models.densenet121(pretrained=True).features
        dense.transition3.pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        modules_dense = list(dense.children())
        self.backdense_0 = torch.nn.Sequential(*modules_dense[:1])
        self.backdense_1 = torch.nn.Sequential(*modules_dense[1:5])
        self.backdense_2 = torch.nn.Sequential(*modules_dense[5:])

        self.c1 = nn.Conv2d(1024, 640, 1)
        self.n1 = nn.BatchNorm2d(640, affine=True)


    def forward(self, image, template_feat):

        x0 = self.backdense_0(image)
        x0_t = self.conv2d_dw_group(x0, template_feat, padding=1)
        x0_cat = x0 + x0_t

        x1 = self.backdense_1(x0_cat)
        x2 = self.backdense_2(x1)
        xf = self.n1 (F.elu(self.c1(x2)))
        return xf

    def conv2d_dw_group(self, x, kernel, padding=0):
        batch, channel = kernel.shape[:2]
        x = x.contiguous().view(1, batch * channel, x.size(2), x.size(3)) # 1 * (b*c) * k * k
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
        out = F.conv2d(x, kernel, groups=batch * channel, padding=padding)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

#Object Attention Branch
class TemplateFeatExtractGlobal(nn.Module):
    def __init__(self, output_dim=1024):
        super(TemplateFeatExtractGlobal, self).__init__()

        self.backbone = models.squeezenet1_1(pretrained=True)
        modules = list(list(self.backbone.children())[0])

        # new conv with 4 channels and already trained weights
        new_conv_0 = nn.Conv2d(4, 64, kernel_size=3, stride=2)
        new_conv_0.weight[:, :3, :, :] = modules[0].weight
        new_conv_0.weight = nn.Parameter(new_conv_0.weight)
        new_conv_0.bias = modules[0].bias
        new_conv_0.bias = nn.Parameter(new_conv_0.bias)
        self.backbone_0 = torch.nn.Sequential(new_conv_0)

        self.backbone_1 = torch.nn.Sequential(*modules[1:5])
        self.backbone_2 = torch.nn.Sequential(*modules[5:])

        self.norm_1 = nn.BatchNorm2d(128, affine=True)
        self.norm_2 = nn.BatchNorm2d(512, affine=True)

        self.final_conv_1 = nn.Conv2d(640, 128, 3)
        self.final_conv_2 = nn.Conv2d(128, 64, 3)

        self.final_norm_1 = nn.BatchNorm2d(128, affine=True)
        self.final_norm_2 = nn.BatchNorm2d(64, affine=True)

    def forward(self, img):
        x0 = self.backbone_0(img)

        x1 = self.backbone_1(x0)
        x2 = self.backbone_2(x1)

        x1_norm = self.norm_1(x1)
        x2_norm = self.norm_2(x2)

        kernel_size = x2.size(3)
        x1_down = F.upsample(x1_norm, size=kernel_size, mode='bilinear')

        xf = torch.cat([x2_norm, x1_down], dim=1)
        xf = self.final_norm_1 (F.elu(self.final_conv_1(xf)))
        xf = self.final_norm_2(F.elu(self.final_conv_2(xf)))

        return xf

#Pose Specific Branch
class TemplateFeatExtract(nn.Module):
    def __init__(self, output_dim=1024):
        super(TemplateFeatExtract, self).__init__()

        self.backbone = models.squeezenet1_1(pretrained=True)
        modules = list(list(self.backbone.children())[0])

        new_conv_0 = nn.Conv2d(4, 64, kernel_size=3, stride=2)
        new_conv_0.weight[:, :3, :, :] = modules[0].weight
        new_conv_0.weight = nn.Parameter(new_conv_0.weight)
        new_conv_0.bias = modules[0].bias
        new_conv_0.bias = nn.Parameter(new_conv_0.bias)
        self.backbone_0 = torch.nn.Sequential(new_conv_0)

        self.backbone_1 = torch.nn.Sequential(*modules[1:5])
        self.backbone_2 = torch.nn.Sequential(*modules[5:])

        self.norm_1 = nn.BatchNorm2d(128, affine=True)
        self.norm_2 = nn.BatchNorm2d(512, affine=True)



    def forward(self, img):

        x0 = self.backbone_0(img)

        x1 = self.backbone_1(x0)
        x2 = self.backbone_2(x1)

        x1_norm = self.norm_1(x1)
        x2_norm = self.norm_2(x2)

        kernel_size = x2.size(3)
        x1_down = F.upsample(x1_norm, size=kernel_size, mode='bilinear')

        xf = torch.cat([x2_norm, x1_down], dim=1)
        return xf




class CorrelationModel(nn.Module):
    def __init__(self, input_dim=1024):
        super(CorrelationModel, self).__init__()

        # additional layers to pose specific branch
        self.c1 = nn.Conv2d(input_dim, input_dim, 3, padding=0)
        self.n1 = nn.BatchNorm2d(input_dim, affine=True)

        self.c2 = nn.Conv2d(input_dim, input_dim, 3, padding=0)
        self.n2 = nn.BatchNorm2d(input_dim, affine=True)

        # correlation layers 
        self.corr_conv_dot = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.norm_corr_dot = nn.BatchNorm2d(256, affine=True)

        self.corr_conv_dot3x3 = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.norm_corr_dot3x3 = nn.BatchNorm2d(256, affine=True)

        self.corr_conv_sub = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.norm_corr_sub = nn.BatchNorm2d(256, affine=True)

        self.cf = nn.Conv2d(768, 512, 3, padding=1)
        self.nf = nn.BatchNorm2d(512, affine=True)

        #segmentation layers
        self.s1 = nn.Conv2d(512, 256, 3, padding=1)
        self.ns1 = nn.BatchNorm2d(256, affine=True)

        self.s2 = nn.Conv2d(256, 128, 3, padding=1)
        self.ns2 = nn.BatchNorm2d(128, affine=True)

        self.s3 = nn.Conv2d(128, 64, 3, padding=1)
        self.ns3 = nn.BatchNorm2d(64, affine=True)

        self.s4 = nn.Conv2d(64, 32, 3, padding=1)
        self.ns4 = nn.BatchNorm2d(32, affine=True)

        self.s5 = nn.Conv2d(32, 16, 3, padding=1)
        self.ns5 = nn.BatchNorm2d(16, affine=True)

        self.seg_final = nn.Conv2d(16, 1, 3, padding=1)

        # Center prediction
        self.corr_conv_heatmap = nn.Conv2d(512, 1, 1)

    def forward(self, image_feat, template_feat, test=False):

        t1 = self.n1(F.elu(self.c1(template_feat)))
        t2 = self.n2(F.elu(self.c2(t1)))
        dot3x3 = self.conv2d_dw_group(image_feat, t2, padding=1)

        avg = F.avg_pool2d(template_feat, 7)
        dot = image_feat * avg
        sub = image_feat - avg

        dot_conv = self.norm_corr_dot(F.elu(self.corr_conv_dot(dot)))
        dot_conv3x3 = self.norm_corr_dot3x3(F.elu(self.corr_conv_dot3x3(dot3x3)))
        sub_conv = self.norm_corr_sub(F.elu(self.corr_conv_sub(sub)))

        #outputs
        x = torch.cat([dot_conv, sub_conv, dot_conv3x3], dim=1)
        x2 = self.nf(F.elu(self.cf(x)))
        
        if not test:

            # Heatmap
            out = self.corr_conv_heatmap(x2)
            heat_map = F.sigmoid(out)

            # segmentation
            s1 = F.upsample(self.ns1(F.elu(self.s1(x2))), scale_factor=2, mode="nearest")
            s2 = F.upsample(self.ns2(F.elu(self.s2(s1))), scale_factor=2, mode="nearest")
            s3 = F.upsample(self.ns3(F.elu(self.s3(s2))), scale_factor=2, mode="nearest")
            s4 = F.upsample(self.ns4(F.elu(self.s4(s3))), size=IMG_SIZE, mode="nearest")
            s5 = self.ns5(F.elu(self.s5(s4)))
            segmentation = self.seg_final(s5) 
            
            return x2, heat_map, segmentation

        return x2, 0, 0

    def conv2d_dw_group(self, x, kernel, padding=0):
        batch, channel = kernel.shape[:2]
        x = x.contiguous().view(1, batch * channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
        out = F.conv2d(x, kernel, groups=batch * channel, padding=padding)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

class Network(NetworkBase):
    def __init__(self):
        super(Network, self).__init__()

        # stage 1 networks
        self.template_feature_extractor_global = TemplateFeatExtractGlobal()
        self.image_feature_extractor = ImageFeatExtract()
        self.template_feature_extractor = TemplateFeatExtract()
        self.correlation_model = CorrelationModel(640)

        # detection networks
        self.anchors = Anchors(pyramid_levels=[4], ratios=[0.5, 1, 2], sizes=[30], scales=[1, 2, 3, 4, 5, 6, 7, 8])
        self.classification = ClassificationModel(512, num_anchors=24)
        self.regression = RegressionModel(512, num_anchors=24)

        # weight init
        prior = 0.01
        self.classification.output.weight.data.fill_(0)
        self.classification.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regression.output.weight.data.fill_(0)
        self.regression.output.bias.data.fill_(0)

        self.correlation_model.corr_conv_heatmap.weight.data.fill_(0)
        self.correlation_model.corr_conv_heatmap.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.correlation_model.seg_final.weight.data.fill_(0)
        self.correlation_model.seg_final.bias.data.fill_(-math.log((1.0 - prior) / prior))

        # utils
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()





    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def compute_template_local(self, img):
        x = self.template_feature_extractor(img)
        return x

    def compute_template_global(self, img):
        x_global = self.template_feature_extractor_global(img)
        return x_global

    def forward(self, image, template, template_mask, global_template, global_template_mask):

        heatmaps_array = []

        template_with_mask = torch.cat([template, template_mask], dim=1)
        global_template_with_mask = torch.cat([global_template, global_template_mask], dim=1)


        # stage 1 - correlations
        template_features_global = self.template_feature_extractor_global(global_template_with_mask)
        features = self.image_feature_extractor(image, template_features_global)
        template_features = self.template_feature_extractor(template_with_mask)
        xcors, heat_map, segmentation = self.correlation_model(features, template_features)

        # stage 2 - detection
        anchors = self.anchors([[xcors.size(2), xcors.size(3)]])
        classifications, _ = self.classification(xcors)
        regression = self.regression(xcors)

        return classifications, regression, anchors, heat_map, segmentation

    def forward_all_templates(self, image, template_features, template_features_global, topk=1):
        with torch.no_grad():

            class_outputs = []
            reg_outputs = []

            # calculate backbone features only once
            template_batch_global = template_features_global[0]
            features = self.image_feature_extractor(image, template_batch_global)


            # Accumulate detections for all local templates
            for i in range(len(template_features)):

                # stage 1 - correlations
                template_batch_local = template_features[i]
                batch_size = template_batch_local.size(0)
                feat_size = features.size()
                features_expand = features.expand([batch_size, feat_size[1], feat_size[2], feat_size[3]])
                xcors_feature, _, _ = self.correlation_model(features_expand, template_batch_local, True)

                #stage 2 - detections
                anchors = self.anchors([[xcors_feature.size(2), xcors_feature.size(3)]])
                classifications, class_heatmap = self.classification(xcors_feature)
                regression = self.regression(xcors_feature)

                if i == 0:
                    class_outputs = classifications
                    reg_outputs = regression
                else:
                    class_outputs = torch.cat([class_outputs, classifications], dim=0)
                    reg_outputs = torch.cat([reg_outputs, regression], dim=0)


            # Keep information about which template is responsible for which predictions
            s = reg_outputs.size()
            obj_indices = []
            for obj_id in range(0, s[0]):
                if obj_id == 0:
                    obj_indices = torch.zeros(1, s[1], 1)
                else:
                    obj_indices = torch.cat([obj_indices, torch.ones(1, s[1], 1) * obj_id], dim=0)


            transformed_anchors = self.regressBoxes(anchors, reg_outputs)
            transformed_anchors = self.clipBoxes(transformed_anchors, image)

            classifications = class_outputs.contiguous().view(1, -1, 2)
            regression = reg_outputs.contiguous().view(1, -1, 2)
            transformed_anchors = transformed_anchors.view(1, -1, 4)
            obj_indices = obj_indices.contiguous().view(1, -1, 1)


            # Keep Top 1000
            maxes = torch.topk(classifications, 1000, dim=1)
            max_score = maxes[0][0, :, 1]
            max_id = maxes[1][0, :, 1]
            anchors_pred = transformed_anchors[0, max_id, :]
            obj_indices = obj_indices[0, max_id, :]

            # NMS with IoU=0.5
            #nms_ids = nms(torch.cat([anchors_pred, max_score.unsqueeze(1)], dim=1), 0.5)
            nms_ids = torchvision.ops.boxes.nms(anchors_pred, max_score, 0.5)
            max_score = max_score[nms_ids]
            max_id = max_id[nms_ids]
            anchors_pred = anchors_pred[nms_ids]
            obj_indices = obj_indices[nms_ids]

            # Keep top "topK"
            max_score = max_score[:topk]
            anchors_pred = anchors_pred[:topk]
            obj_indices = obj_indices[:topk]


            return [max_score, anchors_pred, obj_indices]












