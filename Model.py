import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, mean=0.0, std=0.01)
    elif classname.find('GroupNorm') != -1:
        init.normal_(m.weight.data, 0.1, 0.01)
        init.normal_(m.bias.data, 0.0, 0.01)


class RCB(nn.Module):
    def __init__(self, in_channels):
        super(RCB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(in_channels, in_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.omega1 = nn.Parameter(torch.FloatTensor([0]))
        self.omega2 = nn.Parameter(torch.FloatTensor([0]))
        self.omega3 = nn.Parameter(torch.FloatTensor([0]))
        self.omega4 = nn.Parameter(torch.FloatTensor([0]))
        self.omega5 = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, inputs):
        temp1 = self.conv1(inputs)
        temp2 = self.conv2(temp1)
        temp3 = self.conv3(temp2)
        temp4 = self.conv4(temp3)
        temp5 = self.conv5(temp4)
        out = self.shortcut(
            inputs) + self.omega1 * temp1 + self.omega2 * temp2 + self.omega3 * temp3 + self.omega4 * temp4 + self.omega5 * temp5
        return out


class Mask_Branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mask_Branch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=min(in_channels, out_channels), bias=False),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        out = self.conv1(x)
        return out


class Downsample_Mask(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_Mask, self).__init__()
        self.downsmaple = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.downsmaple(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SSRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSRBlock, self).__init__()

        mid_channels = out_channels
        self.spectral = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, groups=min(in_channels, mid_channels), bias=False),
            nn.GroupNorm(mid_channels, mid_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, groups=min(mid_channels, out_channels), bias=False),
            nn.GroupNorm(out_channels, out_channels),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(mid_channels, mid_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(out_channels, out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(out_channels, out_channels),
        )

    def forward(self, x):
        x = F.leaky_relu(self.shortcut(x) + self.spatial(x) + self.spectral(x), negative_slope=0.1, inplace=True)
        return x


class MixBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixBlock, self).__init__()
        self.SSRB = SSRBlock(in_channels, out_channels)
        self.mask_branch = Mask_Branch(in_channels, out_channels)

    def forward(self, data, mask):
        data = self.SSRB(data)
        mask = self.mask_branch(mask)
        data = data * mask
        return data, mask


class MixedAutoencoder(nn.Module):
    def __init__(self, opt):
        super(MixedAutoencoder, self).__init__()

        self.len_shift = opt.len_shift
        self.nC = opt.bands

        self.conv_in1 = nn.Conv2d(self.nC, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_in2 = nn.Conv2d(self.nC, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out1 = nn.Conv2d(32, self.nC, kernel_size=3, stride=1, padding=1, bias=False)

        self.down1 = MixBlock(32, 32)
        self.downsample1 = Downsample(32, 64)
        self.downsample1_mask = Downsample_Mask(32, 64)
        self.down2 = MixBlock(64, 64)
        self.downsample2 = Downsample(64, 96)
        self.downsample2_mask = Downsample_Mask(64, 96)
        self.down3 = MixBlock(96, 96)
        self.downsample3 = Downsample(96, 128)
        self.downsample3_mask = Downsample_Mask(96, 128)

        self.bottleneck = MixBlock(128, 128)
        self.para_estimator = Para_Estimator(128)

        self.upsample3 = Upsample(128, 96)
        self.up3 = SSRBlock(96 * 2, 96)
        self.upsample2 = Upsample(96, 64)
        self.up2 = SSRBlock(128, 64)
        self.upsample1 = Upsample(64, 32)
        self.up1 = SSRBlock(64, 32)

    def forward(self, r, Phi):
        b, c, h_inp, w_inp = r.shape
        Phi = shift(Phi)
        Phi = Phi[:, :, :, :w_inp]
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        r = F.pad(r, [0, pad_w, 0, pad_h], mode='reflect')
        Phi = F.pad(Phi, [0, pad_w, 0, pad_h], mode='reflect')
        data = self.conv_in1(r)
        mask = self.conv_in2(Phi)

        fea1, mask1 = self.down1(data, mask)
        data = self.downsample1(fea1)
        mask = self.downsample1_mask(mask1)

        fea2, mask2 = self.down2(data, mask)
        data = self.downsample2(fea2)
        mask = self.downsample2_mask(mask2)

        fea3, mask3 = self.down3(data, mask)
        data = self.downsample3(fea3)
        mask = self.downsample3_mask(mask3)

        data, mask = self.bottleneck(data, mask)
        theta = self.para_estimator(data) * 0.001
        data = torch.mul(torch.sign(data), F.relu(torch.abs(data) - theta))

        data = self.upsample3(data)
        data = self.up3(torch.cat([data, fea3], dim=1))

        data = self.upsample2(data)
        data = self.up2(torch.cat([data, fea2], dim=1))

        data = self.upsample1(data)
        data = self.up1(torch.cat([data, fea1], dim=1))

        data = self.conv_out1(data)

        return data[:, :, :h_inp, :w_inp]


class BasicBlock(nn.Module):
    def __init__(self, opt):
        super(BasicBlock, self).__init__()

        self.len_shift = opt.len_shift
        self.nC = opt.bands
        self.rcb = RCB(opt.bands)
        self.Ublock = MixedAutoencoder(opt)

    def forward(self, f, Phi, r):
        rcb = self.rcb(f)
        res_r = self.Ublock(r, Phi)
        f_pred = res_r + rcb + r

        return f_pred


class Para_Estimator(nn.Module):
    def __init__(self, in_nc=28):
        super(Para_Estimator, self).__init__()
        self.fusion = nn.Conv2d(in_nc, in_nc, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(in_nc, in_nc, 3, 2, 1, bias=True, groups=in_nc)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, in_nc, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_nc, in_nc, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_nc, in_nc, 1, padding=0, bias=False),
            nn.Softplus()
        )
        self.bias = nn.Parameter(torch.FloatTensor([.1]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.down_sample(self.relu(self.fusion(x)))
        x = self.avg_pool(x)
        x = self.mlp(x) + self.bias
        return x * 0.1


class Net(torch.nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        netlayer = []
        self.LayerNum = opt.layernum
        self.nC = opt.bands
        self.fusion = nn.Conv2d(self.nC * 2, self.nC, 1, 1, 0)

        for i in range(opt.layernum):
            netlayer.append(BasicBlock(opt))
            netlayer.append(Para_Estimator(28))

        self.net_stage = nn.ModuleList(netlayer)

        self.apply(weights_init_kaiming)

    def forward(self, g, input_mask=None):
        Phi, PhiPhiT = input_mask
        Phi_shift = shift(Phi, len_shift=2)
        normal_mask = torch.sum(Phi_shift, 1)
        normal_mask = normal_mask.unsqueeze(1)
        g_normal = g / normal_mask
        temp_g = g_normal.repeat(1, Phi.shape[1], 1, 1)
        f = g2f(temp_g, len_shift=2, bands=self.nC)
        f = self.fusion(torch.cat([f, f * Phi], dim=1))

        out = []
        for i in range(self.LayerNum):
            Phif = self.mul_Phiz(Phi_shift, f)
            grad = self.mul_PhiTg(Phi_shift, Phif - g)
            rho = self.net_stage[2 * i + 1](grad)
            r = f - rho * grad
            f = self.net_stage[2 * i](f, Phi, r)
            out.append(f)

        return out

    def mul_PhiTg(self, Phi_shift, g):
        temp_1 = g.repeat(1, 28, 1, 1).cuda()
        temp = temp_1 * Phi_shift
        PhiTg = g2f(temp, len_shift=2, bands=self.nC)
        return PhiTg

    def mul_Phiz(self, Phi_shift, z):
        z_shift = shift(z)
        Phiz = Phi_shift * z_shift
        Phiz = torch.sum(Phiz, 1)
        return Phiz.unsqueeze(1)


def g2f(x, len_shift=2, bands=28):
    for i in range(bands):
        x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
    return x[:, :, :, :256]


def shift(x, len_shift=2):
    x = F.pad(x, [0, 54, 0, 0], mode='constant', value=0)
    for i in range(28):
        x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=len_shift * i, dims=2)
    return x



