import torch
from torch import nn
from model.Decoder.attention_Decoder import Decoder
import torch.nn.functional as F
from model.Encoder.uniformer import uniformer_base_ls
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // 4, 1, 1, 0)
        self.conv2 = nn.Conv2d(channel // 4, channel, 1, 1, 0)

    def forward(self, in_f):
        att = nn.AdaptiveAvgPool2d((1, 1))(in_f)
        att = self.conv2(F.relu(self.conv1(att), inplace=True))
        att = nn.Softmax(dim=1)(att)
        att = att - att.min()
        att = att / att.max()
        return att


class BDRM(nn.Module):
    def __init__(self):
        super(BDRM, self).__init__()
        self.cbr1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.cbr2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ca   = CAM(64 * 2)
        self.conv = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(64)

    def forward(self, in_high, in_low, downfactor=2):
        fea_low  = self.cbr1(in_low)
        fea_h2l  = self.cbr2(F.max_pool2d(in_high, kernel_size=downfactor, stride=downfactor))
        fea_ca = torch.cat((fea_low, fea_h2l), dim=1)
        att_ca = self.ca(fea_ca)
        fea_fuse = torch.mul(att_ca, fea_ca)
        fea_out  = F.relu(self.bn(self.conv(fea_fuse)) + in_low, inplace=True)
        return fea_out, att_ca, fea_ca



class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv11 = nn.Conv2d(64, 16, kernel_size=(1, 9), padding=(0, 4))
        self.conv21 = nn.Conv2d(64, 16, kernel_size=(9, 1), padding=(4, 0))
        self.conv12 = nn.Conv2d(16, 1, kernel_size=(9, 1), padding=(4, 0))
        self.conv22 = nn.Conv2d(16, 1, kernel_size=(1, 9), padding=(0, 4))
        self.bn1    = nn.BatchNorm2d(16)
        self.bn2    = nn.BatchNorm2d(16)

    def forward(self, in_f):
        b, c, h, w = in_f.shape
        att1 = self.conv12(F.relu(self.bn1(self.conv11(in_f)), inplace=True))
        att2 = self.conv22(F.relu(self.bn2(self.conv21(in_f)), inplace=True))
        att = att1 + att2
        att = torch.sigmoid(att)
        att = att.view(b, 1, h * w)
        att = nn.Softmax(dim=2)(att)
        att = att - att.min()
        att = att / att.max()
        att = att.view(b, 1, h, w)
        return att

    
def upsample_like(src, tar=None, shape=None):
    if tar is not None:
        src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    elif tar is None and shape is not None:
        src = F.interpolate(src, size=shape, mode='bilinear', align_corners=True)
    return src
    
class TLRM(nn.Module):
    def __init__(self):
        super(TLRM, self).__init__()
        self.cbr1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.cbr2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.sa   = SAM()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(64)

    def forward(self, in_high, in_low):
        fea_low  = self.cbr1(in_low)
        fea_high = self.cbr2(in_high)
        att_low  = self.sa(fea_low)
        att_l2h  = upsample_like(att_low, in_high)
        fea_fuse = fea_high * att_l2h
        fea_out  = F.relu(self.bn(self.conv(fea_fuse)) + in_high, inplace=True)
        return fea_out, att_l2h

class MEA(nn.Module):
    def __init__(self, channel):
        super(MEA, self).__init__()
        
        self.ca1 = ChannelAttention(channel,ratio=2)
        self.ca2 = ChannelAttention(channel,ratio=2)
        self.sa = SpatialAttention()
        self.edg_pred = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.sal_conv = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        s_ca = self.ca1(x) * x
        s_e = self.ca2(x) * x
        e_pred = self.edg_pred(s_e)
        s_mea = self.sal_conv((self.sa(s_ca) + self.sigmoid(e_pred)) * s_ca) +  x

        return s_mea


class ABFRM(nn.Module):
    def __init__(self):
        super(ABFRM, self).__init__()
        self.ressam1 = TLRM()
        self.ressam2 = TLRM()
        self.ressam3 = TLRM()
        
        self.mea1 = MEA(64) 
        self.mea2 = MEA(64)
        self.mea3 = MEA(64)

        self.mea4 = MEA(64) 
        self.mea5 = MEA(64)
        self.mea6 = MEA(64)

        self.rescam1 = BDRM()
        self.rescam2 = BDRM()
        self.rescam3 = BDRM()

    def forward(self, feature):
        f1, f2, f3, f4 = feature[0], feature[1], feature[2], feature[3]
        f1_sa, sa2 = self.ressam3(self.mea4(f1), f2)
        f2_sa, sa3 = self.ressam2(self.mea5(f2), f3)
        f3_sa, sa4 = self.ressam1(self.mea6(f3), f4)



        f2_ca, ca1, ca1f = self.rescam1(self.mea1(f1_sa), f2_sa, 2)
        f3_ca, ca2, ca2f = self.rescam2(self.mea2(f2_sa), f3_sa, 2)
        f4_ca, ca3, ca3f = self.rescam3(self.mea3(f3_sa), f4, 2)
        return (f1_sa, f2_ca, f3_ca, f4_ca)

 


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class MyNet(nn.Module):
    def __init__(self,
                 channel=64):
        super(MyNet, self).__init__()

        backbone = uniformer_base_ls() 
        path = 'model.pth'
        save_model = torch.load(path, map_location='cpu')
        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in save_model['model'].items() if k in model_dict.keys()}
        backbone.load_state_dict(state_dict)
        self.backbone = backbone

        self.ca_1 = ChannelAttention(64)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(128)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(320)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(512)
        self.sa_4 = SpatialAttention()

      

        self.Translayer_1 = BasicConv2d(64, channel, 1)
        self.Translayer_2 = BasicConv2d(128, channel, 1)
        self.Translayer_3 = BasicConv2d(320, channel, 1)
        self.Translayer_4 = BasicConv2d(512, channel, 1)

        self.unn = ABFRM()

        self.Decoder = Decoder(in_channel_List=[64, 64, 64, 64])

    def upsample(self, x, input):
        return F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=True)

    def forward(self, x):
        encoder = self.backbone(x)
        x1 = encoder[0]  
        x2 = encoder[1]  
        x3 = encoder[2]  
        x4 = encoder[3]  

        f1 = self.ca_1(x1) * x1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(x2) * x2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(x3) * x3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(x4) * x4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)
        a,b,c,d = self.unn((f1,f2,f3,f4))

        sal, sig_sal = self.Decoder(a,b,c,d)

        return sal, sig_sal

