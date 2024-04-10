from easydict import EasyDict as edict

# init
__C_PUCPR = edict()

cfg_data = __C_PUCPR

__C_PUCPR.STD_SIZE = (1280,720)
__C_PUCPR.TRAIN_SIZE = (720, 720) # 2D tuple or 1D scalar
__C_PUCPR.DATA_PATH = '/Your/Dataset/Path'

__C_PUCPR.MEAN_STD = ([0.3908707, 0.3613535, 0.36716083], [0.22240958, 0.21731742, 0.21530356])

__C_PUCPR.LABEL_FACTOR = 1
__C_PUCPR.LOG_PARA = 100.
__C_PUCPR.MAX_BBOX = 250
__C_PUCPR.DOWN_RATIO = 1
__C_PUCPR.BBOX = False
__C_PUCPR.CROP = True
__C_PUCPR.BBOXLOSS_WEIGHT = 0.1
__C_PUCPR.DMLOSS_WEIGHT = 100
                        
__C_PUCPR.RESUME_MODEL = ''#model path
__C_PUCPR.TRAIN_BATCH_SIZE = 16 #imgs

__C_PUCPR.VAL_BATCH_SIZE = 14 # must be 1

__C_PUCPR.LOG_DIR = '/Your/Code/Path/log/'
