from __future__ import annotations
import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
import math
from torch.utils import data
from PIL import Image, ImageOps
from datasets.PUCPR.setting import cfg_data
import pandas as pd
import torch.nn.functional as F
# from misc.transforms import RandomRotation
from config import cfg
# from visdom import Visdom
# vis = Visdom(env='main', port=1314)
# vis.line([[0.]], [0], win='train_loss', opts=dict(title='train_loss', legend=['train']))
class PUCPR(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None, depth_transform=None):
        self.data_path = data_path
        self.mode = mode
        self.data_files = self.get_data()
        # self.data_files = [filename for filename in os.listdir(self.img_path) \
        #                    if os.path.isfile(os.path.join(self.img_path, filename))]

        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.depth_transform = depth_transform
        self.max_obj = cfg_data.MAX_BBOX
        self.down_ratio = cfg_data.DOWN_RATIO
        self.guass_kernel = self.GaussianKernel(shape=(15, 15), sigma=4)
        # self.random_rotaion = RandomRotation()

    def __getitem__(self, index):
        # print(index)
        fname = self.data_files[index]
        img, ann = self.read_image_and_gt(fname)
        img = np.array(img)
        if self.main_transform is not None:
            img, ann= self.main_transform(img, ann)
        if self.img_transform is not None:
            img = self.img_transform(img)
        _,h,w = img.shape
        den = torch.zeros((h,w))
        den = self.Addkernel(ann, self.guass_kernel, den)
        oa_gt = torch.zeros((h,w))
        oa_ind = den>0
        oa_gt[oa_ind] = 1.0
        # vis.heatmap(den.unsqueeze(0).flip(0).squeeze(), win='gt',opts=dict(title='gt'))
        # c_g=torch.sum(den)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        # vis.heatmap(den.unsqueeze(0).flip(0).squeeze(), win='gt2',opts=dict(title='gt2'))
        # c_g=torch.sum(den)
        if cfg_data.CROP==True and self.mode=='train':
            img, den, oa_gt = self.random_crop(img, den, oa_gt)

        return img, den, oa_gt, fname

    def get_data(self, ):
        data_file = self.data_path+'/ImageSets/'+self.mode+'.txt'
        data_txt = open(data_file)
        data_names = data_txt.readlines()
        data_files = []
        for data_name in data_names:
            data_name = data_name.split('\n')[0]
            data_files.append(data_name)
        
        return data_files
    def __len__(self):
        return len(self.data_files)

    def read_image_and_gt(self, fname):
        img_file = self.data_path+'/Images/' +fname+'.jpg'
        img = Image.open(img_file)
        if img.mode == 'L':
            img = img.convert('RGB')
        w,h=img.size
        # den = sio.loadmat(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        ann_file = self.data_path+'/Annotations/'+fname+'.txt'
        ann_file = open(ann_file)
        anns = ann_file.readlines()
        gt_points = [[],[]]
        # oa_gt = torch.zeros((h, w))
        if len(anns)!=0:
            for ann in anns:
                ann = ann.split('\n')[0].split(' ')
                ann = [int(x) for x in ann]
                center = [int((ann[2]-ann[0])/2+ann[0]),int((ann[3]-ann[1])/2+ann[1])]
                # oa_gt[ann[1]:ann[3],ann[0]:ann[2]] = 1.0
                gt_points[0].append(center[0])
                gt_points[1].append(center[1])
        return img, gt_points


    def random_crop(self, img, den, oa_gt):
        min_ht = cfg_data.TRAIN_SIZE[0]
        min_wd = cfg_data.TRAIN_SIZE[1]
        _,ht,wd = img.shape
        if ht<min_ht:
            min_ht = ht
        if wd<min_wd:
            min_wd = wd

        X1 = random.randint(0, wd - min_wd)//cfg_data.LABEL_FACTOR*cfg_data.LABEL_FACTOR
        Y1 = random.randint(0, ht - min_ht)//cfg_data.LABEL_FACTOR*cfg_data.LABEL_FACTOR
        X2 = X1 + min_wd
        Y2 = Y1 + min_ht
        # print("Current img: {}, crop index:{}".format(fname, [X1, Y1, X2, Y2]))
        label_x1 = X1//cfg_data.LABEL_FACTOR
        label_y1 = Y1//cfg_data.LABEL_FACTOR
        label_x2 = X2//cfg_data.LABEL_FACTOR
        label_y2 = Y2//cfg_data.LABEL_FACTOR

        img, den, oa_gt = img[:,Y1:Y2,X1:X2], den[label_y1:label_y2,label_x1:label_x2], oa_gt[label_y1:label_y2,label_x1:label_x2]
        return img, den, oa_gt

    def get_num_samples(self):
        return self.num_samples

    def bbox_transform(self, bbox, max_obj, output_w, ann):
        box = len(bbox)
        box_range = min(box, max_obj)
        target_wh = np.zeros((max_obj, 2), dtype=np.float32)
        reg_mask = np.zeros(max_obj, dtype=np.uint8)
        ind = np.zeros(max_obj, dtype=np.int64)
        ann_out = np.zeros((max_obj, 2), dtype=np.float32)
        # ind_print = []
        for i in range(box_range):
            bbox = bbox.float()
            bbox[i] = bbox[i]*self.down_ratio
            w = bbox[i][2] - bbox[i][0]

            h = bbox[i][3] - bbox[i][1]
            center_point = [(bbox[i][2]+bbox[i][0])/2.0, (bbox[i][3] + bbox[i][1])/2.0]
            center_int_point = [math.ceil(center_point[0]), math.ceil(center_point[1])]
            target_wh[i] = 1.*w, 1.*h
            reg_mask[i] = 1
            index = center_int_point[1]*output_w + center_int_point[0]
            ind[i] = index
            if self.mode == 'train':
                if index>0 and index<=cfg_data.TRAIN_SIZE[0]*cfg_data.TRAIN_SIZE[1]:
                    pass
                else:
                    print('index: ', index)
            # ind_print.append(center_int_point[1]*output_w + center_int_point[0])
            ann_out[i] = ann[i]
        # print("Current ind:{}".format(ind_print))
        return target_wh, reg_mask, ind, ann_out

    def Addkernel(self, center_points,guass_kernel,center_den_IPM):
        center_x = center_points[0]
        center_y = center_points[1]
        h, w = center_den_IPM.shape
        for z in range(len(center_x)) :
            cut_x1, cut_x2,cut_y1,cut_y2 = 0, 0, 0, 0
            x, y = center_x[z],center_y[z]
            x1, y1, x2, y2 = x-7,y-7, x+8, y+8
            if x1<0:
                cut_x1 = 0-x1            
                x1 = 0
            if y1<0:
                cut_y1 = 0-y1            
                y1 = 0
            if x2 > w-1:
                cut_x2 = x2-w+1            
                x2 = w-1
            if y2> h-1:
                cut_y2 = y2-h+1            
                y2 = h-1

            # a = center_den_IPM[y1:y2, x1:x2]
            # b = guass_kernel[cut_y1:25-cut_y2,cut_x1:25-cut_x2]
            center_den_IPM[y1:y2, x1:x2]+=guass_kernel[cut_y1:15-cut_y2,cut_x1:15-cut_x2]
        return center_den_IPM
    def GaussianKernel(self, shape=(15, 15), sigma=0.5):
        """
        2D gaussian kernel which is equal to MATLAB's fspecial('gaussian',[shape],[sigma])
        """
        radius_x, radius_y = [(radius-1.)/2. for radius in shape]
        y_range, x_range = np.ogrid[-radius_y:radius_y+1, -radius_x:radius_x+1]
        h = np.exp(- (x_range*x_range + y_range*y_range) / (2.*sigma*sigma))  # (25,25),max()=1~h[12][12]
        h[h < (np.finfo(h.dtype).eps*h.max())] = 0
        max = h.max()
        min = h.min()
        h = (h-min)/(max-min)
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        # a=h.sum()
        return h
        