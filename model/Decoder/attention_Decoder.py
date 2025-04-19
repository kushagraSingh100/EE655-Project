import torch
import torch.nn as nn
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


# FGM 模块
class SSFM(nn.Module):
    def __init__(self, in_dim):
        super(SSFM, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)

        guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out

class ERM(nn.Module):
    def __init__(self, inc, outc):
        super(ERM, self).__init__()
        self.conv_p1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        self.conv_p2 = nn.Conv2d(inc, outc, kernel_size=3,padding = 1)
        self.conv_glb = nn.Conv2d(outc, inc, kernel_size=1)

        self.conv_matt = nn.Sequential(nn.Conv2d(outc, inc, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(inc, inc, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()
    
        self.inc = inc

    def tensor_erode(self, bin_img, ksize=3):
        # padding is added to the original image first to prevent the image size from shrinking after corrosion
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # unfold the original image into a patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # Take the smallest value in each patch
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded

    def tensor_dilate(self, bin_img, ksize=3):  #
        # padding is added to the original image first to prevent the image size from shrinking after corrosion
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # unfold the original image into a patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # Take the largest value in each patch
        dilate = patches.reshape(B, C, H, W, -1)
        dilate, _ = dilate.max(dim=-1)
        return dilate

    def forward(self, x):
        x = self.conv_fuse(x)
        p1 = self.conv_p1(x)
        d = self.tensor_dilate(p1)
        e = self.tensor_erode(p1)
        matt = d - e
        matt = self.conv_matt(matt)

        fea = x * (1 + matt)  # refining
        f = self.conv_p2(fea)
        return f


class UFRM(nn.Module):
    def __init__(self):
        super(UFRM, self).__init__()
        self.deConv_1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, groups=64)
        self.fuseCbr1 = BasicConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.fuseCbr2 = BasicConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, in_f):
        fea1 = F.interpolate(in_f, scale_factor=2, mode='bilinear', align_corners=True)
        fea2 = self.deConv_1(in_f)
        fuse = self.fuseCbr1(fea2)
        fuse = self.fuseCbr2(fuse)
        out = F.relu(self.bn(self.conv(torch.cat((fea2, fuse), dim=1))) + fea1, inplace=True)
        return out

class BiAttention(nn.Module):
    def __init__(self, in_channel):
        super(BiAttention, self).__init__()
        self.conv_h = nn.Linear(in_channel, in_channel)
        self.conv_w = nn.Linear(in_channel, in_channel)
        self.conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel),
                                  nn.ReLU()
                                  )

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N, C, H, W = x.size()
        x_h = x.permute(0, 3, 1, 2).contiguous().view(N * W, -1, H)
        x_w = x.permute(0, 2, 1, 3).contiguous().view(N * H, -1, W)
        x_h_ = self.conv_h(F.avg_pool2d(x, [1, W]).view(N, -1, H).permute(0, 2, 1))
        x_w_ = self.conv_w(F.avg_pool2d(x, [H, 1]).view(N, -1, W).permute(0, 2, 1))
        weight_h = self.softmax(torch.matmul(x_h, x_h_.repeat(W, 1, 1)))
        weight_w = self.softmax(torch.matmul(x_w, x_w_.repeat(H, 1, 1)))
        out_h = torch.bmm(weight_h, x_h).view(N, W, -1, H).permute(0, 2, 3, 1)
        out_w = torch.bmm(weight_w, x_w).view(N, H, -1, W).permute(0, 2, 1, 3)

        out = self.gamma * (out_h + out_w) + x

        return self.conv(out)


class Decoder(nn.Module):
    def __init__(self, in_channel_List=None):
        super(Decoder, self).__init__()

        if in_channel_List is None:
            in_channel_List = [64, 128, 320, 512]



        self.feature_fuse = nn.Sequential(
            BasicConv2d(in_channel_List[3], 1, 3, 1, 1),
        )
        self.u1 = UFRM()
        self.u2 = UFRM()
        self.e1 = ERM(64,64)
        self.e2 = ERM(64,64)
        self.e3 = ERM(64,64)
        self.e4 = ERM(64,64)

        self.b1 = BiAttention(64)
        self.b2 = BiAttention(64)
        self.b3 = BiAttention(64)
        self.b4 = BiAttention(64)

        # self.cgm_4 = FeatureGuideModule(in_channel_List[3])
        # self.cgm_3 = FeatureGuideModule(in_channel_List[2])
        # self.cgm_2 = FeatureGuideModule(in_channel_List[1])

        self.ssfm_4 = SSFM(in_channel_List[3])
        self.ssfm_3 = SSFM(in_channel_List[2])
        self.ssfm_2 = SSFM(in_channel_List[1])

        self.decoder_module3 = BasicConv2d(2*in_channel_List[3] + in_channel_List[2] + in_channel_List[1] , in_channel_List[2], 3, 1, 1)
        self.decoder_module2 = BasicConv2d(2*in_channel_List[2] + in_channel_List[1] + in_channel_List[0], in_channel_List[1], 3, 1, 1)
        self.decoder_module1 = BasicConv2d(2*in_channel_List[1] + in_channel_List[0], in_channel_List[0], 3, 1, 1)

        self.decoder_final = nn.Conv2d(in_channel_List[0], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2, input3, input4):
        size = input1.size()[2:]
        
        m1_2 = F.interpolate(input1, input2.size()[2:], mode='bilinear', align_corners=True)
        m2_1 = F.interpolate(input2, input1.size()[2:], mode='bilinear', align_corners=True)
        m2_3 = F.interpolate(input2, input3.size()[2:], mode='bilinear', align_corners=True)
        m3_2 = F.interpolate(input3, input2.size()[2:], mode='bilinear', align_corners=True)
        m3_4 = F.interpolate(input3, input4.size()[2:], mode='bilinear', align_corners=True)
        m4_3 = F.interpolate(input4, input3.size()[2:], mode='bilinear', align_corners=True)
        
        # input4 = self.e4(input4)
        layer4_1 = F.interpolate(self.u1(input4), size, mode='bilinear', align_corners=True)
        feature_map = self.feature_fuse(layer4_1)

        layer4 = self.ssfm_4(input4, feature_map)
        layer4 = self.b1(layer4)
        feature3 = self.decoder_module3(
            torch.cat([F.interpolate(layer4, scale_factor=2, mode='bilinear', align_corners=True), input3,m4_3,m2_3], 1))

        

        layer3 = self.ssfm_3(feature3, feature_map)
        layer3 = self.b2(layer3)
        feature2 = self.decoder_module2(
            torch.cat([F.interpolate(layer3, scale_factor=2, mode='bilinear', align_corners=True), input2,m1_2,m3_2], 1))

        layer2 = self.ssfm_2(feature2, feature_map)
        layer2 = self.b3(layer2)
        feature1 = self.decoder_module1(
            torch.cat([F.interpolate(layer2, scale_factor=2, mode='bilinear', align_corners=True), input1,m2_1], 1))
        # return feature1
        final_map = F.interpolate(self.decoder_final(self.u2(feature1)), scale_factor=2, mode='bilinear', align_corners=True)
        return final_map, self.sigmoid(final_map)
        # return feature1,layer2,layer3,layer4


if __name__ == '__main__':
    input1 = torch.rand(8, 64, 88, 88)
    input2 = torch.rand(8, 128, 44, 44)
    input3 = torch.rand(8, 320, 22, 22)
    input4 = torch.rand(8, 512, 11, 11)

    cg = Decoder()
    out1, out2 = cg(input1, input2, input3, input4)
    print(out1.shape)
    print(out2.shape)
