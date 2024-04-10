import os

import numpy as np
from scipy.io import savemat
from skimage.metrics import structural_similarity as ssim
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn.functional as F
from models.CC import CrowdCounter
from collections import OrderedDict
# from datasets.SHHB.setting import cfg_data
import cv2
import time
from config import cfg
from misc.utils import *
from tqdm import tqdm, trange
import pdb
from visdom import Visdom
import matplotlib.pyplot as plt
vis = Visdom(env='main', port=8000)
vis.line([[0.]], [0], win='train_loss', opts=dict(title='train_loss', legend=['train']))
vis.line([[0.]], [0], win='dm_loss', opts=dict(title='dm_loss', legend=['train']))
vis.line([[0.]], [0], win='oa_loss', opts=dict(title='oa_loss', legend=['train']))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Trainer():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd 

        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID,self.net_name).cuda()
        params_decay, params_no_decay=self.split_parameters(self.net.CCN)
        self.optimizer = optim.AdamW(params_no_decay, lr=cfg.LR, weight_decay=1e-7)
        # self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=5e-4)
        # self.optimizer = optim.SGD(self.net.CCN.parameters(), cfg.LR, momentum=0.95,weight_decay= 5e-4)
        self.optimizer.add_param_group({'params':params_decay, 'weight_decay': 5e-4})
        del params_decay, params_no_decay
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)          
        # self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=120, eta_min=1e-6)
        self.train_record = {'best_mae': 1e20, 'best_mse':1e20, 'best_model_name': ''}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

        self.epoch = 0
        self.i_tb = 0
        
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        self.train_loader, self.val_loader, self.restore_transform = dataloader()

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)


    def forward(self):
        if not os.path.exists(self.cfg_data.LOG_DIR):
            os.mkdir(self.cfg_data.LOG_DIR)
        file = open(self.cfg_data.LOG_DIR+'training.txt', 'w').close()
        # self.validate_V3()
        best_mae, best_epoch = 1000.0, 0
        for epoch in range(self.epoch,cfg.MAX_EPOCH):
            print("**********************Learing rate in this epoch is {}***********************".format(self.optimizer.param_groups[0]['lr']))
            if cfg.DATASET=='UCF50':
                print('epoch: ', epoch)
            self.epoch = epoch

            # training    
            self.timer['train time'].tic()
            self.train()
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()

            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

            # validation
            if (epoch+1)%cfg.VAL_FREQ==0 or epoch>cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                with torch.no_grad():
                    self.validate_V1()
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )


    def train(self): # training for all datasets
        self.net.train()
        mae, dm_losses, oa_losses, r_loss, d_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        losses = 0.0
        file = open(self.exp_path+'/' + self.exp_name+'/code/log/' + 'training.txt', 'a')
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()

            img, gt_map, oa_gt, fname  = data
            gt_map = gt_map.float()
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            oa_gt = Variable(oa_gt).cuda()
            if cfg.VISUALIZATION:
                vis.heatmap(gt_map[0].flip([0]).squeeze(), win="gt_map", opts=dict(title="gt_map"))
                vis.heatmap(oa_gt[0].flip([0]).squeeze(), win="oa_gt", opts=dict(title="oa_gt"))
                plt.imshow(np.array(self.restore_transform(img[0].detach().cpu())))
                vis.matplot(plt, win="img", opts=dict(title="img"))

            self.optimizer.zero_grad()
            pred_map, oa_pred = self.net(img, gt_map, oa_gt)
            if cfg.VISUALIZATION:
                vis.heatmap(pred_map[0].squeeze().flip([0]), win="pred_map", opts=dict(title="pred_map"))
            dm_loss, oa_loss= self.net.loss
            dm_loss*=1
            oa_loss*=0.01
            dm_losses+=dm_loss.item()
            oa_losses+=oa_loss.item()
            loss = dm_loss+oa_loss
            losses+=loss.item() 
            # loss = self.net.loss
            for j in range(pred_map.shape[0]):
                pred_single = pred_map[j].data.cpu().numpy()
                gt_single = gt_map[j].data.cpu().numpy()
                pred_cnt = np.sum(pred_single)/100
                gt_cnt = np.sum(gt_single)/100

                single_mae = abs(pred_cnt-gt_cnt)
                # print("fname:{}, pred_cnt:{}, gt_cnt:{}, single_mae: {}".format(fname[j], pred_cnt, gt_cnt, single_mae))
                mae+=single_mae
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)

                file.write("[epoch:{}, iter:{}, dm_loss:{:.4f}, oa_loss:{:.4f}, total_loss:{:.4f}, lr:{:.8f}, iter_time:{:.2f}s]\n"\
                        .format(self.epoch+1, i+1, dm_loss.item(), oa_loss.item(), loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff))
                file.write('[cnt: gt: %.1f pred: %.2f]\n' % (gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0][0].sum().data / self.cfg_data.LOG_PARA))

                print("[epoch:{}, iter:{}, dm_loss:{:.4f}, oa_loss:{:.4f}, total_loss:{:.4f}, lr:{:.8f}, iter_time:{:.2f}s]" \
                        .format(self.epoch+1, i+1, dm_loss.item(), oa_loss.item(), loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff))
                print('[cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0][0].sum().data / self.cfg_data.LOG_PARA))
        mae/=(len(self.train_loader)*self.cfg_data.TRAIN_BATCH_SIZE)
        vis.line([[losses/len(self.train_loader)]], [self.epoch], win='train_loss', update='append')
        vis.line([[dm_losses/len(self.train_loader)]], [self.epoch], win='dm_loss', update='append')
        vis.line([[oa_losses/len(self.train_loader)]], [self.epoch], win='oa_loss', update='append')
        file.write("-------------------------------------[Epoch:{}, Train_MAE:{:.2f}]-------------------------------------\n".format(self.epoch, mae))
        print("-------------------------------------[Epoch:{}, Train_MAE:{:.2f}]-------------------------------------".format(self.epoch, mae))       


    def validate_V1(self):# validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()
        # self.net.load_state_dict(torch.load('./epoch0.pth'))
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()
        PSNR, SSIM = AverageMeter(), AverageMeter()
        for vi, data in enumerate(tqdm(self.val_loader), 0):
            if cfg.DATASET == 'SHTRGBD':
                img, gt_map, depth, ann, target_wh, reg_mask, ind, fname = data
            else:
                img, gt_map, oa_gt, fname = data
            # print("validation iter:{}".format(vi))
            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                oa_gt = Variable(oa_gt).cuda()
                if cfg.DATASET == 'SHTRGBD':
                    depth = Variable(depth).cuda()
                    ind = Variable(ind).cuda()
                    target_wh = Variable(target_wh).cuda()
                    reg_mask = Variable(reg_mask).cuda()
                if cfg.VISUALIZATION:
                    vis.heatmap(gt_map[0].flip([0]).squeeze(), win="gt_map", opts=dict(title="gt_map"))
                    vis.heatmap(oa_gt[0].flip([0]).squeeze(), win="oa_gt", opts=dict(title="oa_gt"))
                    plt.imshow(np.array(self.restore_transform(img[0].detach().cpu())))
                    vis.matplot(plt, win="img", opts=dict(title="img"))

                if cfg.DATASET == 'SHTRGBD':
                    pred_map, _ = self.net.forward(img, gt_map, depth, ann, target_wh, reg_mask, ind=None)
                else:
                    pred_map, oa_pred = self.net.forward(img, gt_map, oa_gt)
                if cfg.VISUALIZATION:
                    vis.heatmap(pred_map[0].squeeze().flip([0]), win="pred_map", opts=dict(title="pred_map"))
                    vis.heatmap(gt_map[0].squeeze().flip([0]), win="gt_map", opts=dict(title="gt_map"))
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    if isinstance(self.net.loss, tuple):
                        losses.update(self.net.loss[0].item()+self.net.loss[1].item())
                    else:
                        losses.update(self.net.loss.item())

                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg
        psnr, ssim = PSNR.avg, SSIM.avg
        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, mse, loss],self.train_record)

    def split_parameters(self, module):
        params_decay = []
        params_no_decay = []
        # params_bias = []
        # for k, v in module.named_modules():
        #     if hasattr(v, 'bias') and isinstance(v.bias, nn.parameter):
        #         params_bias.append(v.bias)
        #     if isinstance(v, nn.BatchNorm2d):
        #         params_no_decay.append(v.weight)
        #     elif hasattr(v, 'weight') and isinstance(v.weight, nn.parameter):
        #         params_decay.append(v.weight)
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                params_decay.append(m.weight)
                if m.bias is not None:
                    params_no_decay.append(m.bias)
            elif isinstance(m, torch.nn.modules.conv._ConvNd):
                params_decay.append(m.weight)
                if m.bias is not None:
                    params_no_decay.append(m.bias)
            elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay
    
    def save_feature(self, feature_map_to_save, fname, dir = None, pred=None, gt=None, ext=None):
        feature_map_to_save = (feature_map_to_save - np.min(feature_map_to_save)) / np.ptp(feature_map_to_save) * 255
        feature_map_colored = cv2.applyColorMap(np.uint8(feature_map_to_save), cv2.COLORMAP_JET)
        if pred == None:
            pred = ''
        if gt==None:
            gt=''
        if ext==None:
            ext=''
        if not os.path.exists(dir):
            os.mkdir(dir)
        cv2.imwrite(f'{dir}/{fname}{pred}{gt}{ext}.png', feature_map_colored)


def normalize_feature(feature):
    # 根据特征的范围进行归一化，例如，将其缩放到 [0, 1]
    normalized_feature = (feature - feature.min()) / (feature.max() - feature.min())
    return normalized_feature

def calculate_psnr(feature1, feature2):
    mse = np.mean((feature1 - feature2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # 如果特征被归一化到 [0, 1]，否则根据实际情况设置
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_ssim(feature1, feature2):
    # Convert features to 8-bit (0-255) for SSIM calculation
    feature1_8bit = (feature1 * 255).astype(np.uint8)
    feature2_8bit = (feature2 * 255).astype(np.uint8)

    # Calculate SSIM
    ssim_value, _ = ssim(feature1_8bit, feature2_8bit, full=True)
    return ssim_value

def evaluate(feature1, feature2):
    # 假设 feature1 和 feature2 是两个单通道特征的 NumPy 数组
    # feature1 = np.random.rand(100, 100)  # 临时随机生成示例特征
    # feature2 = np.random.rand(100, 100)  # 临时随机生成示例特征

    # 归一化特征
    normalized_feature1 = normalize_feature(feature1)
    normalized_feature2 = normalize_feature(feature2)

    # 计算PSNR和SSIM
    psnr_value = calculate_psnr(normalized_feature1, normalized_feature2)
    ssim_value = calculate_ssim(normalized_feature1, normalized_feature2)

    print(f"PSNR: {psnr_value} dB")
    print(f"SSIM: {ssim_value}")
    return psnr_value, ssim_value