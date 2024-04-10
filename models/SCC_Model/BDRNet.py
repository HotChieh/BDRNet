from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *
import torchvision.ops
import time
from torchinfo import summary
# model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'
            
            
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, relu=True, bn=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class BDRNet(nn.Module):
    def __init__(self, pretrained=True):
        super(BDRNet, self).__init__()


        self.fuse1_r = nn.Sequential(DeformableConv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fuse2_r = nn.Sequential(DeformableConv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fuse3_r = nn.Sequential(DeformableConv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.c64_1 = nn.Sequential(DeformableConv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.c64_2 = nn.Sequential(DeformableConv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.c64_3 = nn.Sequential(DeformableConv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.ca_64 = ChannelAttention(64)
        self.c32_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.c32_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.c32_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        # self.ca_32 = ChannelAttention(32)


        self.p1 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))
        self.p2 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))
        self.p3 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))
        
        self.oa_p1 = nn.Sequential(DeformableConv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.oa_p2 = nn.Sequential(DeformableConv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.oa_p3 = nn.Sequential(DeformableConv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        # self.oa_multi_att = MultiHeadAttention(query_dim=1, key_dim=1, out_channel=1, num_heads=4)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        # self.p_smooth = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1))
        self.akwa1 = AKWA(32, 1, 2, 3, 4)
        self.akwa2 = AKWA(32, 1, 2, 3, 4)
        self.akwa3 = AKWA(32, 1, 2, 3, 4)
        self.oa_p_smooth = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1), nn.ReLU(inplace=True),
                                         nn.Conv2d(32, 1, kernel_size=1))
        self.cross_att_3 = MultiHeadAttention(query_dim=32, key_dim=32, mid_channel=32, out_channel=1, num_heads=4)
        self.cross_att_2 = MultiHeadAttention(query_dim=32, key_dim=32, mid_channel=32, out_channel=1, num_heads=4)
        self.cross_att_1 = MultiHeadAttention(query_dim=32, key_dim=32, mid_channel=32, out_channel=1, num_heads=4)
        # self.drop=nn.Dropout()
        initialize_weights(self.modules())
        vgg = models.vgg16_bn(pretrained=pretrained)
        # # print(vgg)
        features = list(vgg.features.children())
        self.features1 = nn.Sequential(*features[0:6])
        self.features2 = nn.Sequential(*features[6:13])
        self.features3 = nn.Sequential(*features[13:23])
        self.features4 = nn.Sequential(*features[23:33])
        self.features5 = nn.Sequential(*features[33:43])

    def forward(self, x):
        #Feature Extraction
        x_pre = self.features1(x)  # 64 
        x1 = self.features2(x_pre)   # 128 1/2
        x2 = self.features3(x1)   # 256  1/4
        x3 = self.features4(x2)  # 512  1/8
        x4 = self.features5(x3) # 512 1/16

        x1 = self.fuse1_r(x2)
        x2 = self.fuse2_r(x3)
        x3 = self.fuse3_r(x4)

        x1_64 = self.c64_1(x1)
        x2_64 = self.c64_2(x2)
        x3_64 = self.c64_3(x3)

        x1_32 = self.c32_1(x1_64)
        x2_32 = self.c32_2(x2_64)
        x3_32 = self.c32_3(x3_64)

        #AKWA
        p3 = self.akwa3(x3_32)
        p2 = self.akwa2(x2_32)
        p1 = self.akwa1(x1_32)

        oa_p3 = self.oa_p3(x3_32)
        oa_p2 = self.oa_p2(x2_32)
        oa_p1 = self.oa_p1(x1_32)
        #PwOE
        p3 = self.cross_att_3(query=oa_p3, key=p3)
        p2 = self.cross_att_2(query=oa_p2, key=p2)
        p1 = self.cross_att_1(query=oa_p1, key=p1)        

        p3 = self.p3(p3)
        p2 = self.p2(p2)
        p1 = self.p1(p1)

        p2 = self.up_sample(p3, out_target=p2)+p2
        p1 = self.up_sample(p2, out_target=p1)+p1
        p = self.up_sample(p3, out_target=p1)+self.up_sample(p2, out_target=p1)+p1
        dm = self.up_sample(p, out_target=x)

        oa_p2 = self.up_sample(oa_p3, out_target=oa_p2)+oa_p2
        oa_p1 = self.up_sample(oa_p2, out_target=oa_p1)+oa_p1
        oa_p = self.up_sample(oa_p3, out_target=oa_p1)+self.up_sample(oa_p2, out_target=oa_p1)+oa_p1  
        oa_p = self.oa_p_smooth(oa_p)
        oa_p = self.sig(self.up_sample(oa_p, out_target=x))

        return dm, oa_p

    def up_sample(self, x, out_target):
        _, _, w, h = out_target.size()
        x = F.interpolate(x, size=(w, h), mode='bilinear')
        return x
def make_res_layer(block, planes, blocks, stride=1):

    downsample = None
    inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)  
class AKWA(nn.Module):
    def __init__(self, query_c, d1, d2, d3, d4):
        super().__init__()
        self.query_c = query_c
        self.query1 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=1)
        self.query2 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=1)
        self.query3 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=1)
        self.query4 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=1)

        self.key1 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=3, padding=d1, dilation=d1)
        self.key2 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=3, padding=d2, dilation=d2)
        self.key3 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=3, padding=d3, dilation=d3)
        self.key4 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=3, padding=d4, dilation=d4)


        self.value1 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=1)
        self.value2 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=1)
        self.value3 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=1)
        self.value4 = nn.Conv2d(self.query_c, self.query_c//4, kernel_size=1)

    def forward(self, query):
        b, c, h, w = query.shape
        query1 = self.query1(query).view(b, c//4, -1)
        query2 = self.query2(query).view(b, c//4, -1)
        query3 = self.query3(query).view(b, c//4, -1)
        query4 = self.query4(query).view(b, c//4, -1)

        key1 = self.key1(query).view(b, c//4, -1)
        key2 = self.key2(query).view(b, c//4, -1)
        key3 = self.key3(query).view(b, c//4, -1)
        key4 = self.key4(query).view(b, c//4, -1)

        value1 = self.value1(query).view(b, c//4, -1)
        value2 = self.value2(query).view(b, c//4, -1)
        value3 = self.value3(query).view(b, c//4, -1)
        value4 = self.value4(query).view(b, c//4, -1)

        querys = torch.stack([query1, query2, query3, query4], dim=0)
        keys = torch.stack([key1, key2, key3, key4], dim=0)
        values = torch.stack([value1, value2, value3, value4], dim=0)

        scores = torch.matmul(querys, keys.transpose(2, 3))
        scores = scores / ((self.query_c) ** 0.5)
        scores = F.softmax(scores, dim=3)

        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=2).squeeze(0)  # [N, T_q, num_units]
        out = out.view(b, c, h, w)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class DeformableConv2d(nn.Module):
    def __init__(self,
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False):
        super(DeformableConv2d, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                        2 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        # modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        # self.regular_conv.weight = self.regular_conv.weight.half() if x.dtype == torch.float16 else \
        #     self.regular_conv.weight
        x = torchvision.ops.deform_conv2d(input=x.float(),
                                            offset=offset.float(),
                                            weight=self.regular_conv.weight,
                                            bias=self.regular_conv.bias,
                                            padding=(self.padding, self.padding),
                                            # mask=modulator,
                                            stride=self.stride,
                                            )
        return x


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''
 
    def __init__(self, query_dim, key_dim, mid_channel,out_channel, num_heads):
 
        super().__init__()
        self.num_units = mid_channel
        self.num_heads = num_heads
        self.key_dim = key_dim
 
        self.W_query = nn.Conv2d(query_dim, self.num_units,kernel_size=1, stride=1)
        self.W_key = nn.Conv2d(key_dim, self.num_units,kernel_size=1, stride=1)
        self.W_value = nn.Conv2d(key_dim, self.num_units,kernel_size=1, stride=1)
        # self.out_conv = nn.Conv2d(self.num_units, out_channel,kernel_size=1, stride=1)
    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)
        b, c, h, w = values.shape
        querys = querys.view(querys.shape[0], querys.shape[1], -1)
        keys = keys.view(keys.shape[0], keys.shape[1], -1)
        values = values.view(values.shape[0], values.shape[1], -1)


        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=1), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=1), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=1), dim=0)  # [h, N, T_k, num_units/h]
 
        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
 
        ## mask
        if mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads,1,querys.shape[2],1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)
 
        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=2).squeeze(0)  # [N, T_q, num_units]
        out = out.view(b, c, h, w)
        # out = self.out_conv(out)
        return out
    

if __name__ == "__main__":
    model = BDRNet()
    tmp_0 = model(torch.rand(1, 3, 512, 512).cuda())
    print(tmp_0.shape)

    summary(model, (1, 3, 512, 512))# summary的函数内部参数形式与导入的第三方库有关，否则报错
