from calendar import c
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
import pdb


class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()        
        self.model_name = model_name
        if model_name == 'AlexNet':
            from .SCC_Model.AlexNet import AlexNet as net        
        elif model_name == 'MCNN':
            from .SCC_Model.MCNN import MCNN as net           
        elif model_name == 'BDRNet':
            from  .SCC_Model.BDRNet import BDRNet as net
        # '''test params'''
        # from torchinfo import summary
        # model = net().cuda()
        # tmp_0 = model(torch.rand(1, 3, 512, 512).cuda())
        # # print(tmp_0[0].shape)
        # summary(model, (1, 3, 512, 512))# summary的函数内部参数形式与导入的第三方库有关，否则报错
        # '''test done'''
        self.CCN = net()
        print(self.CCN)
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        # self.l1 = nn.L1Loss(reduce=True, size_average=True).cuda()
        # self.loss_ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 10])).cuda()
        self.bce = DiceBCELoss().cuda()
        self.iou_loss = IoULoss().cuda()
    @property
    def loss(self):
        if cfg.DATASET=='SHTRGBD':
            if self.loss_wh is not None:
                return self.loss_mse, self.loss_r, self.loss_d, self.loss_wh, self.sim_loss
            else:
                return self.loss_mse, self.loss_r, self.loss_d
        # elif cfg.NET=='DDIPMN':
        #     return self.loss_mse
        else:
            return self.loss_mse, self.BCE_Dice_loss+self.IoU_loss
    def PearsonCorrelation(self, tensor_1,tensor_2):
        x = tensor_1
        y = tensor_2

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost
    def regL1loss(self, wh_pred, mask, target):
        mask = mask.unsqueeze(2).expand_as(wh_pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(wh_pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        # ind_check = ind>0
        # if len(ind_check.nonzero())!=0:
        #     print("index is more than matrix!, the number is {}, matrix size is {}".format(ind[ind_check.nonzero()], feat.size(1)))
        #     print("feat:{}, ind:{}".format(feat, ind))
        # print("index :{},".format(ind[ind_check.nonzero()]))
        feat = self._gather_feat(feat, ind)
        return feat
    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        max_ind = torch.max(ind)
        try:
            feat = feat.gather(1, ind)
        except:
            print("ind :{}, feat: {}.".format(ind, feat.size(1)))
            raise
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def forward(self, img, gt_map, oa_gt, depth=None, ann=None, target_wh=None, reg_mask = None, ind = None):
        if depth is not None:
            density_map, r_map, d_map, wh_out = self.CCN(img, depth)
            # wh_pred = torch.zeros_like(target_wh)
            # self.loss_wh = None
            if ind is not None:
                wh_pred = self._transpose_and_gather_feat(wh_out, ind)
                self.loss_wh = self.build_bbox_loss(wh_pred, reg_mask, target_wh)
            else:
                self.loss_wh = None
            self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
            self.loss_r = self.build_loss(r_map.squeeze(), gt_map.squeeze())
            self.loss_d = self.build_loss(d_map.squeeze(), gt_map.squeeze())
            # self.loss_sa = self.build_loss(sa_map.squeeze(), gt_map.squeeze())
            sa_loss = self.PearsonCorrelation(density_map.squeeze(), gt_map.squeeze()).cuda()
            self.sim_loss = 1.0-sa_loss
            return density_map, wh_out
        else:
            density_map,oa_pred = self.CCN(img)
            self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
            self.BCE_Dice_loss = self.bce(oa_pred.squeeze(), oa_gt.squeeze())
            self.IoU_loss = self.iou_loss(oa_pred.squeeze(), oa_gt.squeeze())
            return density_map, oa_pred

    def build_loss(self, density_map, gt_data):
        # cos_loss = self.cosine_loss(density_map, gt_data)
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse
    def build_bbox_loss(self, wh_pred, reg_mask, target_wh):
        loss_wh = self.regL1loss(wh_pred, reg_mask, target_wh)
        return loss_wh
    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map
    def cosine_loss(self, density_map, gt_data):
        B, _, _ = density_map.size()
        cos_mean = (density_map+gt_data)/2.0
        cos_correct = torch.cosine_similarity((density_map-cos_mean).view(B, -1), (gt_data-cos_mean).view(B, -1), dim = 1)
        cos = (1.0-cos_correct).cuda()
        return torch.mean(cos)
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        pred = torch.zeros_like(inputs).cuda()
        pred_ind_pos = (inputs > 0.5)
        pred[pred_ind_pos] = torch.tensor(1.0).cuda()
        tgt = targets.bool()
        pred = pred.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()

        # intersect, union, intersect / union if (union > 0) else 1.0
        return torch.tensor(1-(intersect / union if (union > 0) else 1.0)).cuda()