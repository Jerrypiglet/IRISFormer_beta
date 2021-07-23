import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from icecream import ic

from models_def.model_nvidia.AppGMM_adaptive import SSNFeatsTransformAdaptive

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
        self.if_feat_recon_adaptive = self.opt.cfg.MODEL_GMM.feat_recon_adaptive.enable

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

        if self.opt.cfg.MODEL_GMM.enable and self.opt.cfg.MODEL_GMM.feat_recon_adaptive.enable:
            # print(x1.shape, x2.shape, x3.shape) # torch.Size([8, 64, 120, 160]) torch.Size([8, 128, 60, 80]) torch.Size([8, 256, 30, 40])
            self.ssn_x3 = SSNFeatsTransformAdaptive(self.opt, (3, 4))
            self.ssn_x2 = SSNFeatsTransformAdaptive(self.opt, (6, 8))
            self.ssn_x1 = SSNFeatsTransformAdaptive(self.opt, (12, 16))
            # self.ssn_x1 = SSNFeatsTransformAdaptive(self.opt, (6, 8))


    def forward(self, x):
        extra_output_dict = {}

        x1 = F.relu(self.gn1(self.conv1(self.pad1(x))), True)
        if self.if_feat_recon_adaptive and 'x1' in self.opt.cfg.MODEL_GMM.feat_recon_adaptive.layers_list:
            x1_ssn = self.ssn_x1(x1)
            x1 = x1_ssn['feats_recon']
            extra_output_dict['x1_affinity'] = x1_ssn['abs_affinity']

        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1))), True)
        if self.if_feat_recon_adaptive and 'x2' in self.opt.cfg.MODEL_GMM.feat_recon_adaptive.layers_list:
            x2_ssn = self.ssn_x2(x2)
            x2 = x2_ssn['feats_recon']
            extra_output_dict['x2_affinity'] = x2_ssn['abs_affinity']

        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2))), True)
        if self.if_feat_recon_adaptive and 'x3' in self.opt.cfg.MODEL_GMM.feat_recon_adaptive.layers_list:
            x3_ssn = self.ssn_x3(x3)
            x3 = x3_ssn['feats_recon']
            extra_output_dict['x3_affinity'] = x3_ssn['abs_affinity']

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
        return x1, x2, x3, x4, x5, x6, extra_output_dict

class decoder0(nn.Module):
    def __init__(self, opt, mode=-1, modality='', out_channel=3, input_dict_guide=None,  if_PPM=False):
        super(decoder0, self).__init__()
        self.opt = opt
        self.mode = mode
        self.modality = modality

        self.if_PPM = if_PPM

        self.if_feat_recon_adaptive = self.opt.cfg.MODEL_GMM.feat_recon_adaptive.enable

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

        self.dpadFinal = nn.ReplicationPad2d(1)

        dconv_final_in_channels = 64
        if self.if_PPM:
            dconv_final_in_channels = 128

        self.dconvFinal = nn.Conv2d(in_channels=dconv_final_in_channels, out_channels=out_channel, kernel_size = 3, stride=1, bias=True)

        if self.opt.cfg.MODEL_GMM.enable and self.opt.cfg.MODEL_GMM.feat_recon_adaptive.enable:
            self.ssn_dx3 = SSNFeatsTransformAdaptive(opt, (3, 4))
            self.ssn_dx4 = SSNFeatsTransformAdaptive(opt, (6, 8))
            # self.ssn_dx5 = SSNFeatsTransformAdaptive(opt, (12, 16))
            self.ssn_dx5 = SSNFeatsTransformAdaptive(opt, (6, 8))
            # self.ssn_dx6 = SSNFeatsTransformAdaptive(opt, (12, 16))

        self.flag = True

    def forward(self, im, x1, x2, x3, x4, x5, x6, input_extra_dict=None):
        extra_output_dict = {}

        dx1 = F.relu(self.dgn1(self.dconv1(x6 ) ) )
        # if if_appearance_recon:
        #     dx1 = input_extra_dict['MODEL_GMM'].appearance_recon(input_extra_dict['gamma_GMM'], dx1, scale_feat_map=32)
        xin1 = torch.cat([dx1, x5], dim = 1)
        # if if_appearance_recon:
        #     xin1 = input_extra_dict['MODEL_GMM'].appearance_recon(input_extra_dict['gamma_GMM'], xin1, scale_feat_map=32)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear') ) ), True)

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear')
        # if if_appearance_recon:
        #     dx2 = input_extra_dict['MODEL_GMM'].appearance_recon(input_extra_dict['gamma_GMM'], dx2, scale_feat_map=16)
        xin2 = torch.cat([dx2, x4], dim=1 )
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear') ) ), True)
        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear')
        if self.if_feat_recon_adaptive and 'dx3' in self.opt.cfg.MODEL_GMM.feat_recon_adaptive.layers_list:
            dx3_ssn = self.ssn_dx3(dx3)
            dx3 = dx3_ssn['feats_recon']
            extra_output_dict['dx3_affinity'] = dx3_ssn['abs_affinity']

        xin3 = torch.cat([dx3, x3], dim=1)
        # if if_appearance_recon:
        #     xin3 = input_extra_dict['MODEL_GMM'].appearance_recon(input_extra_dict['gamma_GMM'], xin3, scale_feat_map=8)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear') ) ), True)
        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear')
        if self.if_feat_recon_adaptive and 'dx4' in self.opt.cfg.MODEL_GMM.feat_recon_adaptive.layers_list:
            dx4_ssn = self.ssn_dx4(dx4)
            dx4 = dx4_ssn['feats_recon']
            extra_output_dict['dx4_affinity'] = dx4_ssn['abs_affinity']

        xin4 = torch.cat([dx4, x2], dim=1 )
        # if if_appearance_recon:
        #     xin4 = input_extra_dict['MODEL_GMM'].appearance_recon(input_extra_dict['gamma_GMM'], xin4, scale_feat_map=4)
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear') ) ), True)
        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear')
        if self.if_feat_recon_adaptive and 'dx5' in self.opt.cfg.MODEL_GMM.feat_recon_adaptive.layers_list:
            dx5_ssn = self.ssn_dx5(dx5)
            dx5 = dx5_ssn['feats_recon']
            extra_output_dict['dx5_affinity'] = dx5_ssn['abs_affinity']

        xin5 = torch.cat([dx5, x1], dim=1 )
        # if  :
        #     xin5 = input_extra_dict['MODEL_GMM'].appearance_recon(input_extra_dict['gamma_GMM'], xin5, scale_feat_map=2)
        dx6 = F.relu(self.dgn6(self.dconv6(F.interpolate(xin5, scale_factor=2, mode='bilinear') ) ), True)

        im_trainval_RGB_mask_pooled_mean = None

        if dx6.size(3) != im.size(3) or dx6.size(2) != im.size(2):
            dx6 = F.interpolate(dx6, [im.size(2), im.size(3)], mode='bilinear')
        # if self.if_feat_recon_adaptive:
            # dx6 = self.ssn_dx6(dx6)['feats_recon']

        # print(dx4.shape, dx5.shape, dx6.shape) # torch.Size([4, 128, 60, 80]) torch.Size([4, 64, 120, 160]) torch.Size([4, 64, 240, 320])
        if self.if_PPM:
            dx6 = self.ppm(dx6)


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
