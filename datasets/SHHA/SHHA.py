import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
from misc.transforms import RandomRotation
import pandas as pd
import math
import torchvision.transforms as transforms
from config import cfg

class SHHA(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img_new'
        self.gt_path = data_path + '/den_new'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
        self.main_transform = main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform     
        self.random_rotaion = RandomRotation()
    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)      
        # img = self.resize(img, dataset_name="SHA")
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 
        if self.img_transform is not None:
            img = self.img_transform(img)         
        if self.gt_transform is not None:
            den = self.gt_transform(den)           
        img, den = self.random_rotaion(img, den)
        h, w = den.shape
        oa_gt = torch.zeros((h,w))
        oa_ind = den>0
        oa_gt[oa_ind] = 1.0
        return img, den, oa_gt, fname

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        
        den = den.astype(np.float32, copy=False)    
        den = Image.fromarray(den)  
        return img, den    

    def get_num_samples(self):
        return self.num_samples       
            
    def resize(self, img, dataset_name):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width
        if dataset_name == "SHA":
            if resize_height <= 416:
                tmp = resize_height
                resize_height = 416
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 416:
                tmp = resize_width
                resize_width = 416
                resize_height = (resize_width / tmp) * resize_height
            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        else:
            raise NameError("Only SHA is released")
        img = transforms.Resize([resize_height, resize_width])(img)
        return img