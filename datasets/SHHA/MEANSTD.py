import os
import re
import numpy as np
import cv2
from tqdm import tqdm
train_img_path = '/data1/haojie/shanghaitech_part_A/train/img_new/'
test_img_path = '/data1/haojie/shanghaitech_part_A/test/img_new/'

train_num = 300
test_num = 182

means = [0, 0, 0]
stdevs = [0, 0, 0]
for i in tqdm(range(train_num)):
    img = cv2.imread(train_img_path+str(i+1)+".jpg")
    img = np.asarray(img)
    img = img.astype(np.float32) / 255
    for j in range(3):
        means[j] += img[:, :, j].mean()
        stdevs[j] += img[:, :, j].std()
        
for m in tqdm(range(test_num)):
    img = cv2.imread(test_img_path+str(m+1)+".jpg")
    img = np.asarray(img)
    img = img.astype(np.float32) / 255
    for n in range(3):
        means[n] += img[:, :, n].mean()
        stdevs[n] += img[:, :, n].std()     

means.reverse()
stdevs.reverse()

means = np.asarray(means) / (train_num+test_num)
stdevs = np.asarray(stdevs) / (train_num+test_num)
print("normMean = {};normStd = {}".format(means, stdevs))