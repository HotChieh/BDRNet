from easydict import EasyDict as edict

# init
__C_SHHB = edict()

cfg_data = __C_SHHB
__C_SHHB.LOG_DIR = '/data1/haojie/C-3-Framework/log/'
__C_SHHB.STD_SIZE = (768,1024)
__C_SHHB.TRAIN_SIZE = (768,768)
__C_SHHB.DATA_PATH = '/data1/haojie/SHHB/'               
__C_SHHB.DMLOSS_WEIGHT = 100
__C_SHHB.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_SHHB.LABEL_FACTOR = 1
__C_SHHB.LOG_PARA = 100.

__C_SHHB.RESUME_MODEL = ''#model path
__C_SHHB.TRAIN_BATCH_SIZE =  6 #imgs

__C_SHHB.VAL_BATCH_SIZE = 6 # 


