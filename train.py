import os
import numpy as np
import torch

from config import cfg

#------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)>=1:
    torch.cuda.set_device(cfg.MAIN_GPU[0])

torch.backends.cudnn.benchmark = True


#------------prepare data loader------------
data_mode = cfg.DATASET
if data_mode == 'SHHA':
    from datasets.SHHA.loading_data import loading_data 
    from datasets.SHHA.setting import cfg_data 
elif data_mode == 'SHHB':
    from datasets.SHHB.loading_data import loading_data 
    from datasets.SHHB.setting import cfg_data 
elif data_mode == 'PUCPR':
    from datasets.PUCPR.loading_data import loading_data
    from datasets.PUCPR.setting import cfg_data
#------------Prepare Trainer------------
net = cfg.NET
if net in ['MCNN', 'AlexNet', 'BDRNet']:
    from trainer import Trainer
elif net in ['SANet']: 
    from trainer_for_M2TCC import Trainer # double losses but signle output
elif net in ['CMTL']: 
    from trainer_for_CMTL import Trainer # double losses and double outputs
elif net in ['PCCNet']:
    from trainer_for_M3T3OCC import Trainer

#------------Start Training------------
gpu_env = ",".join(str(x) for x in gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_env

pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(loading_data,cfg_data,pwd)
cc_trainer.forward()
