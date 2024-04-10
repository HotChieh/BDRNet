from easydict import EasyDict as edict

# init
__C_SHHA = edict()

cfg_data = __C_SHHA
__C_SHHA.LOG_DIR = '/data/haojie//CODE/EXTRemoteCC/log/'
__C_SHHA.STD_SIZE = (768,1024)
# __C_SHHA.TRAIN_SIZE = (576,768) # 2D tuple or 1D scalar
__C_SHHA.TRAIN_SIZE = (416,416) # 2D tuple or 1D scalar 61.82， 101.13
# __C_SHHA.TRAIN_SIZE = (576,768) 
__C_SHHA.DATA_PATH = '/data/haojie/DATASETS/shanghaitech_part_A/'               

# __C_SHHA.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
# __C_SHHA.MEAN_STD = ([0.40861232, 0.36843685, 0.35842451], [0.26376192, 0.25389463, 0.253246]) #Gao's dataset 自测
__C_SHHA.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #ORG PGC's dataset
# __C_SHHA.MEAN_STD=([0.4087144 , 0.36838419, 0.35831454],[0.26562682, 0.25546091, 0.25487697]) #ORG 自测

__C_SHHA.LABEL_FACTOR = 1
__C_SHHA.LOG_PARA = 100.

__C_SHHA.RESUME_MODEL = ''#model path
__C_SHHA.TRAIN_BATCH_SIZE = 2 #imgs

__C_SHHA.VAL_BATCH_SIZE = 2 # must be 1


