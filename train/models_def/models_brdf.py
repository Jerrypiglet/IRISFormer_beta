import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# import pac
# import pac_simplified as pac
from models_def.model_matseg import logit_embedding_to_instance
from icecream import ic

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

def LSregress(pred, gt, origin):
    nb = pred.size(0)
    origSize = pred.size()
    pred = pred.reshape(nb, -1)
    gt = gt.reshape(nb, -1)

    coef = (torch.sum(pred * gt, dim = 1) / torch.clamp(torch.sum(pred * pred, dim=1), min=1e-5)).detach()
    coef = torch.clamp(coef, 0.001, 1000)
    for n in range(0, len(origSize) -1):
        coef = coef.unsqueeze(-1)
    pred = pred.view(origSize)

    predNew = origin * coef.expand(origSize)

    return predNew

def LSregressDiffSpec(diff, spec, imOrig, diffOrig, specOrig):
    nb, nc, nh, nw = diff.size()
    
    # Mask out too bright regions
    mask = (imOrig < 0.9).float() 
    diff = diff * mask 
    spec = spec * mask 
    im = imOrig * mask

    diff = diff.view(nb, -1)
    spec = spec.view(nb, -1)
    im = im.view(nb, -1)

    a11 = torch.sum(diff * diff, dim=1)
    a22 = torch.sum(spec * spec, dim=1)
    a12 = torch.sum(diff * spec, dim=1)

    frac = a11 * a22 - a12 * a12
    b1 = torch.sum(diff * im, dim = 1)
    b2 = torch.sum(spec * im, dim = 1)

    # Compute the coefficients based on linear regression
    coef1 = b1 * a22  - b2 * a12
    coef2 = -b1 * a12 + a11 * b2
    coef1 = coef1 / torch.clamp(frac, min=1e-2)
    coef2 = coef2 / torch.clamp(frac, min=1e-2)

    # Compute the coefficients assuming diffuse albedo only
    coef3 = torch.clamp(b1 / torch.clamp(a11, min=1e-5), 0.001, 1000)
    coef4 = coef3.clone() * 0

    frac = (frac / (nc * nh * nw)).detach()
    fracInd = (frac > 1e-2).float()

    coefDiffuse = fracInd * coef1 + (1 - fracInd) * coef3
    coefSpecular = fracInd * coef2 + (1 - fracInd) * coef4

    for n in range(0, 3):
        coefDiffuse = coefDiffuse.unsqueeze(-1)
        coefSpecular = coefSpecular.unsqueeze(-1)

    coefDiffuse = torch.clamp(coefDiffuse, min=0, max=1000)
    coefSpecular = torch.clamp(coefSpecular, min=0, max=1000)

    diffScaled = coefDiffuse.expand_as(diffOrig) * diffOrig
    specScaled = coefSpecular.expand_as(specOrig) * specOrig 

    # Do the regression twice to avoid clamping
    renderedImg = torch.clamp(diffScaled + specScaled, 0, 1)
    renderedImg = renderedImg.view(nb, -1)
    imOrig = imOrig.view(nb, -1)

    coefIm = (torch.sum(renderedImg * imOrig, dim = 1) \
            / torch.clamp(torch.sum(renderedImg * renderedImg, dim=1), min=1e-5)).detach()
    coefIm = torch.clamp(coefIm, 0.001, 1000)
    
    coefIm = coefIm.view(nb, 1, 1, 1)

    diffScaled = coefIm * diffScaled 
    specScaled = coefIm * specScaled

    return diffScaled, specScaled


class encoder0(nn.Module):
    def __init__(self, opt, cascadeLevel = 0, isSeg = False, in_channels = 3, encoder_exclude=[]):

        super(encoder0, self).__init__()
        self.isSeg = isSeg
        self.opt = opt
        self.cascadeLevel = cascadeLevel

        self.encoder_exclude = encoder_exclude + self.opt.cfg.MODEL_BRDF.encoder_exclude

        self.pad1 = nn.ReplicationPad2d(1)
        if self.cascadeLevel == 0:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, bias =True)
        else:
            self.conv1 = nn.Conv2d(in_channels=17, out_channels = 64, kernel_size =4, stride =2, bias = True)

        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=64)

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=16, num_channels=256)

        if 'x5' not in self.encoder_exclude:
            self.pad5 = nn.ZeroPad2d(1)
            self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True)
            self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512)

        if 'x6' not in self.encoder_exclude:
            self.pad6 = nn.ZeroPad2d(1)
            self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True)
            self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024)

    def forward(self, x):
        x1 = F.relu(self.gn1(self.conv1(self.pad1(x))), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1))), True)
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2))), True)
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3))), True)
        
        if 'x5' not in self.encoder_exclude:
            x5 = F.relu(self.gn5(self.conv5(self.pad5(x4))), True)
        else:
            x5 = x1

        if 'x6' not in self.encoder_exclude:
            x6 = F.relu(self.gn6(self.conv6(self.pad6(x5))), True)
        else:
            x6 = x1

        # print(x.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape) # [16, 3, 192, 256, ]) [16, 64, 96, 128, ]) [16, 128, 48, 64,) [16, 256, 24, 32,) [16, 256, 12, 16,) [16, 512, 6, 8],  [16, 1024, 6, 8]
        return x1, x2, x3, x4, x5, x6, {}


class decoder0_guide(nn.Module):
    def __init__(self, opt, mode=-1, out_channel=3, in_C = [1024, 1024, 512, 512, 256, 128], out_C = [512, 256, 256, 128, 64, 64], group_C = [32, 16, 16, 8, 4, 4],  if_PPM=False):
        super(decoder0_guide, self).__init__()
        self.opt = opt
        self.mode = mode
        self.if_PPM = if_PPM

        self.if_matseg_guide = self.opt.cfg.MODEL_MATSEG.enable and self.opt.cfg.MODEL_MATSEG.if_guide
        self.if_matseg_guide_layers = ['dconv1', 'dconv2', 'dconv3', 'dconv4', 'dconv5', 'dconv6']

        # self.guide_C = self.opt.cfg.MODEL_MATSEG.guide_channels
        self.if_semseg_guide = self.opt.cfg.MODEL_SEMSEG.enable and self.opt.cfg.MODEL_SEMSEG.if_guide
        self.if_semseg_guide_layers = ['dconv3', 'dconv4', 'dconv5']

        assert not(self.if_matseg_guide and self.if_semseg_guide)

        self.dconv1 = self.get_conv(name='dconv1', in_channels=in_C[0], out_channels=out_C[0], kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=group_C[0], num_channels=out_C[0])

        self.dconv2 = self.get_conv(name='dconv2', in_channels=in_C[1], out_channels=out_C[1], kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=group_C[1], num_channels=out_C[1])

        self.dconv3 = self.get_conv(name='dconv3', in_channels=in_C[2], out_channels=out_C[2], kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=group_C[2], num_channels=out_C[2])

        self.dconv4 = self.get_conv(name='dconv4', in_channels=in_C[3], out_channels=out_C[3], kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=group_C[3], num_channels=out_C[3])

        self.dconv5 = self.get_conv(name='dconv5', in_channels=in_C[4], out_channels=out_C[4], kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=group_C[4], num_channels=out_C[4])

        self.dconv6 = self.get_conv(name='dconv6', in_channels=in_C[5], out_channels=out_C[5], kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn6 = nn.GroupNorm(num_groups=group_C[5], num_channels=out_C[5])

        fea_dim = out_C[5]
        if self.if_PPM:
            bins=(1, 2, 3, 6)
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels=fea_dim, out_channels=out_channel, kernel_size = 3, stride=1, bias=True)

    def get_conv(self, name, in_channels, out_channels, kernel_size, stride, bias, padding):
        if (self.if_matseg_guide and name in self.if_matseg_guide_layers) or (self.if_semseg_guide and name in self.if_semseg_guide_layers):
            return pac.PacConv2d(in_channels, out_channels, kernel_size, padding=padding)
        else:
            return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding = padding, bias=bias)

    def interpolate_x_to_y(self, x, y):
        if x.size(3) != y.size(3) or x.size(2) != y.size(2):
            # print(x.shape, y.shape)
            return F.interpolate(x, [y.size(2), y.size(3)], mode='bilinear')
        else:
            return x

    def forward(self, im, x1, x2, x3, x4, x5, x6, input_dict_extra={}):
        if 'input_dict_guide' in input_dict_extra:
            input_dict_guide = input_dict_extra['input_dict_guide']
        else:
            input_dict_guide = None

        if self.if_matseg_guide:
            xin0 = self.dconv1(x6, self.interpolate_x_to_y(input_dict_guide['p5'], x6))
        else:
            xin0 = self.dconv1(x6)
        dx1 = F.relu(self.dgn1(xin0))

        # print('BRDF x6', x6.shape)


        xin1 = F.interpolate(torch.cat([dx1, x5], dim = 1), scale_factor=2, mode='bilinear')
        # print('BRDF xin1', xin1.shape)
        # print(xin1.shape, input_dict_guide['p5'].shape)
        if self.if_matseg_guide:
            # print(xin1.shape, input_dict_guide['p4'].shape)
            xin1 = self.dconv2(xin1, self.interpolate_x_to_y(input_dict_guide['p4'], xin1))
        else:
            xin1 = self.dconv2(xin1)
        dx2 = F.relu(self.dgn2(xin1), True)


        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear')
        xin2 = F.interpolate(torch.cat([dx2, x4], dim=1), scale_factor=2, mode='bilinear')
        # print('BRDF xin2', xin2.shape)
        if self.if_matseg_guide:
            xin2 = self.dconv3(xin2, self.interpolate_x_to_y(input_dict_guide['p3'], xin2))
        elif self.if_semseg_guide:
            featinx2 = F.interpolate(input_dict_guide['x2'], [xin2.size(2), xin2.size(3)], mode='bilinear')
            if self.opt.cfg.MODEL_BRDF.if_debug_arch:
                print(input_dict_guide['x2'].shape, '---> + ', xin2.shape)
            xin2 = self.dconv3(xin2, featinx2)
        else:
            xin2 = self.dconv3(xin2)
        dx3 = F.relu(self.dgn3(xin2), True)


        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear')
        xin3 = F.interpolate(torch.cat([dx3, x3], dim=1), scale_factor=2, mode='bilinear')
        # print('BRDF xin3', xin3.shape)
        if self.if_matseg_guide:
            xin3 = self.dconv4(xin3, self.interpolate_x_to_y(input_dict_guide['p2'], xin3))
        elif self.if_semseg_guide:
            featinx3 = F.interpolate(input_dict_guide['x1'], [xin3.size(2), xin3.size(3)], mode='bilinear')
            if self.opt.cfg.MODEL_BRDF.if_debug_arch:
                print(input_dict_guide['x1'].shape, '---> + ', xin3.shape)
            xin3 = self.dconv4(xin3, featinx3)
        else:
            xin3 = self.dconv4(xin3)
        dx4 = F.relu(self.dgn4(xin3), True)


        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear')
        xin4 = F.interpolate(torch.cat([dx4, x2], dim=1), scale_factor=2, mode='bilinear')
        # print('BRDF xin4', xin4.shape)
        if self.if_matseg_guide:
            xin4 = self.dconv5(xin4, self.interpolate_x_to_y(input_dict_guide['p1'], xin4))
        elif self.if_semseg_guide:
            featinx4 = F.interpolate(input_dict_guide['x0_2'], [xin4.size(2), xin4.size(3)], mode='bilinear')
            if self.opt.cfg.MODEL_BRDF.if_debug_arch:
                print(input_dict_guide['x0_2'].shape, '---> + ', xin4.shape)
            xin4 = self.dconv5(xin4, featinx4)
        else:
            xin4 = self.dconv5(xin4)
        dx5 = F.relu(self.dgn5(xin4), True)


        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear')
        xin5 = F.interpolate(torch.cat([dx5, x1], dim=1), scale_factor=2, mode='bilinear')
        # print('BRDF xin5', xin5.shape)
        if self.if_matseg_guide:
            xin5 = self.dconv6(xin5, self.interpolate_x_to_y(input_dict_guide['p0'], xin5))
        else:
            xin5 = self.dconv6(xin5)
        dx6 = F.relu(self.dgn6(xin5), True)


        if dx6.size(3) != im.size(3) or dx6.size(2) != im.size(2):
            dx6 = F.interpolate(dx6, [im.size(2), im.size(3)], mode='bilinear')
        if self.if_PPM:
            dx6 = self.ppm(dx6)

        x_orig = self.dconvFinal(self.dpadFinal(dx6))

        # print(x1, x2, x3, x4, x5, x6)
        # print(dx1.shape, dx2.shape, dx3.shape, dx4.shape, dx5.shape, dx6.shape, x_orig.shape)

        if self.mode == 0:
            x_out = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
        elif self.mode == 1:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1)).expand_as(x_orig)
            x_out = x_orig / torch.clamp(norm, min=1e-6)
        elif self.mode == 2:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        elif self.mode == 3:
            x_out = F.softmax(x_orig, dim=1)
        elif self.mode == 4:
            x_orig = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
        else:
            x_out = x_orig
        return x_out

class decoder0(nn.Module):
    def __init__(self, opt, mode=-1, modality='', out_channel=3, input_dict_guide=None,  if_PPM=False):
        super(decoder0, self).__init__()
        self.opt = opt
        self.mode = mode
        self.modality = modality
        
        self.if_PPM = if_PPM

        self.if_appearance_recon = False
        if self.opt.cfg.MODEL_GMM.enable and self.opt.cfg.MODEL_GMM.appearance_recon.enable and self.modality in self.opt.cfg.MODEL_GMM.appearance_recon.modalities:
            self.if_appearance_recon = True

        self.if_albedo_pooling = self.opt.cfg.MODEL_MATSEG.if_albedo_pooling
        self.if_albedo_asso_pool_conv = self.opt.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv
        self.if_albedo_pac_pool = self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool
        
        self.dconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv4 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dconv5 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dconv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn6 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.relu = nn.ReLU(inplace = True )

        fea_dim = 64
        if self.if_PPM:
            bins=(1, 2, 3, 6)
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2


        if self.if_albedo_asso_pool_conv:
            self.acco_pool_1 = pac.PacPool2d(kernel_size=3, stride=1, padding=1, dilation=1)
            self.acco_pool_2 = pac.PacPool2d(kernel_size=3, stride=1, padding=2, dilation=2)
            self.acco_pool_4 = pac.PacPool2d(kernel_size=3, stride=1, padding=4, dilation=4)

        # self.acco_pool_mean = pac.PacPool2d(kernel_size=3, stride=1, padding=4, dilation=4)

        if self.if_albedo_pac_pool:
            self.acco_pool_mean_list = torch.nn.ModuleList([])
            self.albedo_pac_pool_mean_layers = self.opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers

            # self.acco_pool_mean_list.append(
            #     pac.PacPool2d(kernel_size=7, stride=1, padding=30, dilation=10, normalize_kernel=True)
            # )


            # self.acco_pool_mean_list.append(
            #     pac.PacPool2d(kernel_size=5, stride=1, padding=20, dilation=10, normalize_kernel=True)
            # )

            self.acco_pool_mean_list += [
                pac.PacPool2d(kernel_size=3, stride=1, padding=20, dilation=20, normalize_kernel=True), 
                pac.PacPool2d(kernel_size=3, stride=1, padding=10, dilation=10, normalize_kernel=True), 
                pac.PacPool2d(kernel_size=3, stride=1, padding=5, dilation=5, normalize_kernel=True), 
                # pac.PacPool2d(kernel_size=3, stride=1, padding=2, dilation=2, normalize_kernel=True), 
                # pac.PacPool2d(kernel_size=3, stride=1, padding=1, dilation=1, normalize_kernel=True), 
            ]


            # self.acco_pool_mean = pac.PacPool2d(kernel_size=7, stride=1, padding=30, dilation=10, normalize_kernel=True)
            # self.acco_pool_mean = pac.PacPool2d(kernel_size=7, stride=1, padding=15, dilation=5, normalize_kernel=True)
            # self.acco_pool_mean = pac.PacPool2d(kernel_size=7, stride=1, padding=6, dilation=2, normalize_kernel=True)
            # self.acco_pool_mean = pac.PacPool2d(kernel_size=3, stride=1, padding=20, dilation=20, normalize_kernel=True)
            # self.acco_pool_mean = pac.PacPool2d(kernel_size=3, stride=1, padding=10, dilation=10, normalize_kernel=True)
            # self.acco_pool_mean = pac.PacPool2d(kernel_size=3, stride=1, padding=5, dilation=5, normalize_kernel=True)
            # self.acco_pool_mean = pac.PacPool2d(kernel_size=3, stride=1, padding=2, dilation=2, normalize_kernel=True)
            # self.acco_pool_mean = pac.PacPool2d(kernel_size=3, stride=1, padding=1, dilation=1, normalize_kernel=True)
            # self.acco_pool_mean = pac.PacPool2d(kernel_size=5, stride=1, padding=20, dilation=10, normalize_kernel=True)
            # self.acco_pool_mean = pac.PacPool2d(kernel_size=5, stride=1, padding=10, dilation=5, normalize_kernel=True)



        self.dpadFinal = nn.ReplicationPad2d(1)

        dconv_final_in_channels = 64
        if self.if_PPM:
            dconv_final_in_channels = 128
        if self.if_albedo_pooling:
            dconv_final_in_channels = 128
        if self.if_albedo_asso_pool_conv:
            dconv_final_in_channels = 64 * 4
        if self.if_albedo_pac_pool:
            if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_mean:
                dconv_final_in_channels = 64 * 2 if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_keep_input else 64
            else:
                dconv_final_in_channels = 64 * (1 + len(self.acco_pool_mean_list)) if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_keep_input else 64 * len(self.acco_pool_mean_list)
        assert not(self.if_albedo_pooling and self.if_albedo_asso_pool_conv), 'self.if_albedo_pooling and self.if_albedo_asso_pool_conv cannot be True at the same time!'

        self.dconvFinal = nn.Conv2d(in_channels=dconv_final_in_channels, out_channels=out_channel, kernel_size = 3, stride=1, bias=True)

        self.flag = True

    def mask_pooled_mean(self, dx6, instance, num_mat_masks_batch):
        dx6_pooled_mean = torch.zeros_like(dx6)
        # c = dx6.shape[1]
        for batch_id, (dx6_single, instance_single, num_mat) in enumerate(zip(dx6, instance, num_mat_masks_batch)):
            num_mat = num_mat.item()
            for mat_id in range(num_mat):
                instance_mat = instance_single[mat_id].bool() # [h, w]
                # dx6_single_mat_selected = torch.masked_select(dx6_single, instance_mat).view(c, -1) # [c, ?]
                dx6_single_mat_selected = dx6_single[:, instance_mat] # [c, ?]
                # dx6_single_mat_selected_mean = torch.mean(dx6_single_mat_selected, 1, keepdim=True) # [c, 1]
                # dx6_single_mat_selected_mean_expand = dx6_single_mat_selected_mean.expand(-1, torch.sum(instance_mat).item()).reshape(-1)
                # dx6_pooled_mean[batch_id][instance_mat.expand(c, -1, -1)] -= dx6_single_mat_selected_mean_expand
                # dx6_pooled_mean[batch_id][:, instance_mat] -= 1
                dx6_pooled_mean[batch_id][:, instance_mat] = torch.mean(dx6_single_mat_selected, 1, keepdim=True)
        return dx6_pooled_mean

    def forward(self, im, x1, x2, x3, x4, x5, x6, input_dict_extra=None):

        extra_output_dict = {}

        if self.if_albedo_pooling:
            if self.opt.cfg.MODEL_MATSEG.albedo_pooling_from == 'gt':
                instance = input_dict_extra['matseg-instance']
                num_mat_masks_batch = input_dict_extra['semseg-num_mat_masks_batch']
            else:
                matseg_logits = input_dict_extra['matseg-logits']
                matseg_embeddings = input_dict_extra['matseg-embeddings']
                mat_notlight_mask_cpu = input_dict_extra['mat_notlight_mask_cpu']
                instance, num_mat_masks_batch, predict_segmentation = logit_embedding_to_instance(mat_notlight_mask_cpu, matseg_logits, matseg_embeddings, self.opt)
            # print(instance.shape, num_mat_masks_batch.shape) # torch.Size([16, 50, 240, 320]) torch.Size([16])
            # if self.flag:
            #     np.save('instance.npy', instance.cpu().numpy())
            #     np.save('num_mat_masks_batch.npy', num_mat_masks_batch.cpu().numpy())
            #     self.flag = False

        if self.if_albedo_asso_pool_conv or self.if_albedo_pac_pool:
            matseg_embeddings = input_dict_extra['matseg-embeddings']
            
        dx1 = F.relu(self.dgn1(self.dconv1(x6 ) ) )
        if self.if_appearance_recon:
            dx1 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], dx1, scale_feat_map=32)
        xin1 = torch.cat([dx1, x5], dim = 1)
        # if self.if_appearance_recon:
        #     xin1 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], xin1, scale_feat_map=32)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear') ) ), True)

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear')
        if self.if_appearance_recon:
            dx2 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], dx2, scale_feat_map=16)
        xin2 = torch.cat([dx2, x4], dim=1 )
        # if self.if_appearance_recon:
        #     xin2 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], xin2, scale_feat_map=16)
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear') ) ), True)

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear')
        if self.if_appearance_recon:
            dx3 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], dx3, scale_feat_map=8)
        xin3 = torch.cat([dx3, x3], dim=1)
        # if self.if_appearance_recon:
        #     xin3 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], xin3, scale_feat_map=8)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear') ) ), True)

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear')
        if self.if_appearance_recon:
            dx4 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], dx4, scale_feat_map=4)
        xin4 = torch.cat([dx4, x2], dim=1 )
        # if self.if_appearance_recon:
        #     xin4 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], xin4, scale_feat_map=4)
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear') ) ), True)

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear')
        if self.if_appearance_recon:
            dx5 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], dx5, scale_feat_map=2)
        xin5 = torch.cat([dx5, x1], dim=1 )
        # if  :
        #     xin5 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], xin5, scale_feat_map=2)
        dx6 = F.relu(self.dgn6(self.dconv6(F.interpolate(xin5, scale_factor=2, mode='bilinear') ) ), True)

        im_trainval_RGB_mask_pooled_mean = None
        if self.if_albedo_pooling:
            dx6_pooled_mean = self.mask_pooled_mean(dx6, instance, num_mat_masks_batch)
            dx6 = torch.cat([dx6, dx6 - dx6_pooled_mean], 1)

            if self.opt.cfg.MODEL_MATSEG.albedo_pooling_debug:
                im_trainval_RGB_mask_pooled_mean = self.mask_pooled_mean(input_dict_extra['im_trainval_RGB'], instance, num_mat_masks_batch)

        if self.if_albedo_asso_pool_conv:
            dx6_pool_1 = self.acco_pool_1(dx6, matseg_embeddings)
            dx6_pool_2 = self.acco_pool_2(dx6, matseg_embeddings)
            dx6_pool_4 = self.acco_pool_4(dx6, matseg_embeddings)
            # print(dx6_pool_1.shape, dx6_pool_2.shape, dx6_pool_4.shape)
            dx6 = torch.cat([dx6, dx6_pool_1, dx6_pool_2, dx6_pool_4], 1)

        if self.if_albedo_pac_pool:
            dx6_pool_mean_list = []
            for acco_pool_mean in self.acco_pool_mean_list:
                dx6_pool_mean = acco_pool_mean(dx6, matseg_embeddings)
                dx6_pool_mean_list.append(dx6_pool_mean)
            if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_mean:
                dx6_pool_mean = torch.stack(dx6_pool_mean_list, dim=0).mean(dim=0)
            else:
                dx6_pool_mean = torch.cat(dx6_pool_mean_list, dim=1)

            # print(dx6_pool_mean.shape)
            # dx6 = torch.cat([dx6, dx6_pool_1, dx6_pool_2, dx6_pool_4], 1)
            if self.opt.cfg.MODEL_MATSEG.albedo_pooling_debug:
                im_trainval_RGB_mask_pooled_mean_list = []
                for acco_pool_mean in self.acco_pool_mean_list:
                    im_trainval_RGB_mask_pooled_mean = acco_pool_mean(input_dict_extra['im_trainval_RGB'], matseg_embeddings * (2. * input_dict_extra['mat_notlight_mask_gpu_float'] - 1))
                    im_trainval_RGB_mask_pooled_mean_list.append(im_trainval_RGB_mask_pooled_mean)
                im_trainval_RGB_mask_pooled_mean = torch.stack(im_trainval_RGB_mask_pooled_mean_list, dim=0).mean(dim=0)


            if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_keep_input            :
                dx6 = torch.cat([dx6, dx6_pool_mean], 1)
            else:
                dx6 = dx6_pool_mean

        if dx6.size(3) != im.size(3) or dx6.size(2) != im.size(2):
            dx6 = F.interpolate(dx6, [im.size(2), im.size(3)], mode='bilinear')

        if self.if_PPM:
            # ic(dx6.shape)
            dx6 = self.ppm(dx6)
            # ic(dx6.shape)

        # if self.if_appearance_recon:
        #     dx6 = input_dict_extra['MODEL_GMM'].appearance_recon(input_dict_extra['gamma_GMM'], dx6, scale_feat_map=1)

        x_orig = self.dconvFinal(self.dpadFinal(dx6 ) )
        # ic(x_orig.shape)
        # print(x1, x2, x3, x4, x5, x6)
        
        # print(x6.shape, dx1.shape, dx2.shape, dx3.shape, dx4.shape, dx5.shape, dx6.shape, x_orig.shape) 
        # torch.Size([2, 1024, 7, 10]) torch.Size([2, 512, 7, 10]) torch.Size([2, 256, 15, 20]) torch.Size([2, 256, 30, 40]) torch.Size([2, 128, 60, 80]) torch.Size([2, 64, 120, 160]) torch.Size([2, 64, 240, 320]) torch.Size([2, 3, 240, 320])


        if self.mode == 0:
            x_out = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
        elif self.mode == 1:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1) ).expand_as(x_orig)
            x_out = x_orig / torch.clamp(norm, min=1e-6)
        elif self.mode == 2:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        elif self.mode == 3:
            x_out = F.softmax(x_orig, dim=1)
        elif self.mode == 4:
            x_orig = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
        elif self.mode == 5: # clip to 0., inf
            x_out = self.relu(torch.mean(x_orig, dim=1).unsqueeze(1))
        elif self.mode == 6: # sigmoid to 0., 1. -> inverse to 0., inf
            x_out = torch.sigmoid(torch.mean(x_orig, dim=1).unsqueeze(1))
            x_out = 1. / (x_out + 1e-6)
        else:
            x_out = x_orig

        return_dict = {'x_out': x_out, 'extra_output_dict': extra_output_dict}
        # if self.if_albedo_pooling:
        return_dict.update({'im_trainval_RGB_mask_pooled_mean': im_trainval_RGB_mask_pooled_mean})

        return return_dict
