import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED =1012 # random seed,  for reproduction
__C.DATASET = 'PUCPR' # datasets 

__C.NET = 'BDRNet' # net selection

__C.PRE_GCC = False # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = 'path to model' # path to model

__C.VISUALIZE = True
__C.BN = True
__C.PRETRAINED =False
# contine training
__C.VISUALIZATION = False
__C.RESUME =False
__C.RESUME_PATH = '/data2/haojie/CODE/BDRNet/exp/04-08_20-24_PUCPR_BDRNet_0.0001/latest_state.pth' # 
__C.RESUME_BEST =False
__C.RESUME_BEST_PATH = './XXX/your_best.pth'


__C.GPU_ID = [2,3] # sigle gpu: [0], [1] ...; multi gpus: [0,1]
__C.MAIN_GPU = [2]
# learning rate settings
__C.LR = 1e-3# learning rate 
__C.LR_DECAY = 0.995# decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency
__C.MAX_EPOCH = 500

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-3# SANet:0.001 CMTL 0.0001
                                                               

# print 
__C.PRINT_FREQ = 1

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)

__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes


#------------------------------VAL------------------------
__C.VAL_DENSE_START = 2
__C.VAL_FREQ = 1 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes



#================================================================================
#================================================================================
#================================================================================  
