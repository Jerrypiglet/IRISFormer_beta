import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# import pac
import pac_simplified as pac
from models_def.model_matseg import logit_embedding_to_instance

class decoder0_pacpool(nn.Module):
    def __init__(self, opt, mode=-1, out_channel=3, input_dict_guide=None):
        super(decoder0_pacpool, self).__init__()
        self.mode = mode
        self.opt = opt

        # from utils.utils_misc import only1true
        # assert only1true([self.if_albedo_pooling, self.if_albedo_asso_pool_conv, self.if_albedo_pac_pool]), 'Only ONE of the pooling methods canbe true at the same time!'

        # self.acco_pool_mean_list = torch.nn.ModuleList([])
        self.albedo_pac_pool_mean_layers = self.opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers

        # self.acco_pool_mean_list = torch.nn.ModuleList([
        #     pac.PacPool2d(kernel_size=3, stride=1, padding=20, dilation=20, normalize_kernel=True), 
        #     pac.PacPool2d(kernel_size=3, stride=1, padding=10, dilation=10, normalize_kernel=True), 
        #     pac.PacPool2d(kernel_size=3, stride=1, padding=5, dilation=5, normalize_kernel=True), 
        #     # pac.PacPool2d(kernel_size=3, stride=1, padding=2, dilation=2, normalize_kernel=True), 
        #     # pac.PacPool2d(kernel_size=3, stride=1, padding=1, dilation=1, normalize_kernel=True), 
        # ])
        self.pool_layers_num = {name: None for name in self.opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers_allowed}

        if 'x6' in self.albedo_pac_pool_mean_layers:
            self.x6_pac_pool, self.x6_pac_pool_len = self.build_pac_pool_list('x6', kernel_sizes=[], paddings=[], dilations=[])
        self.dconv1 = nn.Conv2d(in_channels=self.get_in_c(1024, 'x6', self.pool_layers_num['x6']), out_channels=512, kernel_size=3, stride=1, padding = 1, bias=True) #in: [16, 512, 7, 10]
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512 )

        if 'xin1' in self.albedo_pac_pool_mean_layers:
            self.xin1_pac_pool, self.xin1_pac_pool_len = self.build_pac_pool_list('xin1', kernel_sizes=[], paddings=[], dilations=[])
        self.dconv2 = nn.Conv2d(in_channels=self.get_in_c(1024, 'xin1', self.pool_layers_num['xin1']), out_channels=256, kernel_size=3, stride=1, padding = 1, bias=True) #in: [16, 512, 7, 10]
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        if 'xin2' in self.albedo_pac_pool_mean_layers:
            self.xin2_pac_pool, self.xin2_pac_pool_len = self.build_pac_pool_list('xin2', kernel_sizes=[], paddings=[], dilations=[])
        self.dconv3 = nn.Conv2d(in_channels=self.get_in_c(512, 'xin2', self.pool_layers_num['xin2']), out_channels=256, kernel_size=3, stride=1, padding=1, bias=True) #in: [16, 256, 15, 20]
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        if 'xin3' in self.albedo_pac_pool_mean_layers:
            self.xin3_pac_pool, self.xin3_pac_pool_len = self.build_pac_pool_list('xin3', kernel_sizes=[3, 3, 3], paddings=[5, 2, 1], dilations=[5, 2, 1])
            # self.xin3_pac_pool, self.xin3_pac_pool_len = self.build_pac_pool_list(kernel_sizes=[7], paddings=[6], dilations=[2])
        self.dconv4 = nn.Conv2d(in_channels=self.get_in_c(512, 'xin3', self.pool_layers_num['xin3']), out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True) #in: [16, 256, 30, 40]
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=128 )

        if 'xin4' in self.albedo_pac_pool_mean_layers:
            # self.xin4_pac_pool, self.xin4_pac_pool_len = self.build_pac_pool_list(kernel_sizes=[7], paddings=[9], dilations=[3])
            self.xin4_pac_pool, self.xin4_pac_pool_len = self.build_pac_pool_list('xin4', kernel_sizes=[3, 3, 3], paddings=[10, 5, 2], dilations=[10, 5, 2])
        self.dconv5 = nn.Conv2d(in_channels=self.get_in_c(256, 'xin4', self.pool_layers_num['xin4']), out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True) # in: [16, 128, 60, 80]
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=64 )

        if 'xin5' in self.albedo_pac_pool_mean_layers:
            # self.xin5_pac_pool, self.xin5_pac_pool_len = self.build_pac_pool_list(kernel_sizes=[7], paddings=[15], dilations=[5])
            self.xin5_pac_pool, self.xin5_pac_pool_len = self.build_pac_pool_list('xin5', kernel_sizes=[3, 3, 3], paddings=[20, 10, 5], dilations=[20, 10, 5])
        self.dconv6 = nn.Conv2d(in_channels=self.get_in_c(128, 'xin5', self.pool_layers_num['xin5']), out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True) # in: [16, 64, 120, 160]
        self.dgn6 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dpadFinal = nn.ReplicationPad2d(1)
        if 'xin6' in self.albedo_pac_pool_mean_layers:
            # self.xin6_pac_pool, self.xin6_pac_pool_len = self.build_pac_pool_list(kernel_sizes=[3, 3, 3], paddings=[20, 10, 5], dilations=[20, 10, 5])
            self.xin6_pac_pool, self.xin6_pac_pool_len = self.build_pac_pool_list('xin6', kernel_sizes=[7], paddings=[30], dilations=[10])
        self.dconvFinal = nn.Conv2d(in_channels=self.get_in_c(64, 'xin6', self.pool_layers_num['xin6']), out_channels=3, kernel_size = 3, stride=1, bias=True) # in: [16, 64, 240, 320]

        self.flag = True

        if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_debug_deform:
            self.demo_kH = self.demo_kW = 7
            self.demo_padding = 30
            self.demo_dilation = 10
            # self.demo_padding = 19
            # self.demo_dilation = 5

            self.conv_offset_op = nn.Conv2d(3, 2 * self.demo_kH * self.demo_kW,
                    kernel_size = (self.demo_kH, self.demo_kW),
                    stride = (1, 1),
                    padding = self.demo_padding,
                    dilation = self.demo_dilation, 
                    bias = True)

            self.conv_offset_op.weight.data.fill_(0.0)
            self.conv_offset_op.bias.data.fill_(0.0)

            self.pac_pool_deform_op = pac.PacPool2d(kernel_size=self.demo_kH, stride=1, padding=self.demo_padding, dilation=self.demo_dilation, normalize_kernel=True, native_impl=False, \
                if_deform=True)

    def build_pac_pool_list(self, layer_name, kernel_sizes=[], strides=[], paddings=[], dilations=[], native_impl=False):
        assert len(kernel_sizes) == len(paddings) == len(dilations)
        assert len(kernel_sizes) != 0
        if strides == []:
            strides = [1] * len(kernel_sizes)
        assert len(kernel_sizes) == len(strides)

        if layer_name is not None:
            self.pool_layers_num[layer_name] = len(kernel_sizes)

        # if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_deform and layer_name in self.opt.cfg.MODEL_MATSEG.albedo_pac_pool_deform_layers:
        #     if_deform = True
        # else:
        #     if_deform = False
            
        return_module_list = torch.nn.ModuleList([
            pac.PacPool2d(kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, normalize_kernel=True, native_impl=native_impl, if_deform=False) \
                for kernel, stride, padding, dilation in zip(kernel_sizes, strides, paddings, dilations)])


        #     conv_offset_op = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
        #                 kernel_size=(kH, kW),
        #                 stride=(1, 1),
        #                 padding=padding,
        #                 dilation=dilation, 
        #                 bias=True)

        return return_module_list, len(kernel_sizes)

    
    def get_in_c(self, in_c, layer_name, pac_pool_len=None):
        assert layer_name in self.opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers_allowed
        out_c = in_c
        if layer_name in self.albedo_pac_pool_mean_layers:
            if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_mean:
                out_c = in_c * 2 if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_keep_input else in_c
            else:
                assert pac_pool_len is not None
                out_c = in_c * (1 + pac_pool_len) if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_keep_input else in_c * pac_pool_len
        else:
            out_c = in_c
            
        return out_c

    def pac_pool_transform(self, x, matseg_embeddings_mat_notlight_mask_gpu_float, x_pac_pool, force_mean=False, return_kernel_list=False, offsets=None):
        matseg_embeddings, mat_notlight_mask_gpu_float = matseg_embeddings_mat_notlight_mask_gpu_float
        if matseg_embeddings.size(3) != x.size(3) or matseg_embeddings.size(2) != x.size(2):
            matseg_embeddings_use = F.interpolate(matseg_embeddings, [x.size(2), x.size(3)], mode='bilinear')
            mat_notlight_mask_gpu_float_use = F.interpolate(mat_notlight_mask_gpu_float, [x.size(2), x.size(3)], mode='nearest')
        else:
            matseg_embeddings_use = matseg_embeddings
            mat_notlight_mask_gpu_float_use = mat_notlight_mask_gpu_float

        pac_pool_mean_list = []
        kernel_list = []
        for pac_pool_mean in x_pac_pool:
            pac_pool_mean, kernel = pac_pool_mean(x, matseg_embeddings_use * (2. * mat_notlight_mask_gpu_float_use - 1), offsets=offsets)
            pac_pool_mean_list.append(pac_pool_mean)
            kernel_list.append(kernel)
        if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_mean or force_mean:
            pac_pool_out = torch.stack(pac_pool_mean_list, dim=0).mean(dim=0)
        else:
            pac_pool_out = torch.cat(pac_pool_mean_list, dim=1)

        if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_keep_input and not force_mean:
            x_out = torch.cat([x, pac_pool_out], 1)
        else:
            x_out = pac_pool_out
        
        if return_kernel_list:
            return x_out, kernel_list
        else:
            return x_out


    def forward(self, im, x1, x2, x3, x4, x5, x6, input_extra_dict=None):

        return_dict = {}

        matseg_embeddings = input_extra_dict['matseg-embeddings']
        # matseg_embeddings = matseg_embeddings * (2. * input_extra_dict['mat_notlight_mask_gpu_float'] - 1)
        mat_notlight_mask_gpu_float = input_extra_dict['mat_notlight_mask_gpu_float']

        im_trainval_RGB_mask_pooled_mean, kernel_list = None, None

        if_deform = True
        native_impl = True

        if self.opt.cfg.MODEL_MATSEG.albedo_pooling_debug and self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_debug_deform:
            im_in = input_extra_dict['im_trainval_RGB']
            offsets = self.conv_offset_op(im_in)
            # offsets = torch.zeros((B, kH*kW*2, H, W), device=im_in.device)
            # offsets = offsets + 50
            # offsets = torch.zeros_like(offsets, device=offsets.device, requires_grad=True) + 10
            offsets[:, 1::2, :, :] = offsets[:, 1::2, :, :] + 100
            im_trainval_RGB_mask_pooled_mean, kernel_list = self.pac_pool_deform_op(im_in, matseg_embeddings * (2. * mat_notlight_mask_gpu_float  - 1), offsets=offsets)
            print(im_trainval_RGB_mask_pooled_mean.shape, '======')


        if self.opt.cfg.MODEL_MATSEG.albedo_pooling_debug and not self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool_debug_deform:
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[3, 3, 3], paddings=[30, 20, 10], dilations=[30, 20, 10])
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[3, 3, 3], paddings=[20, 10, 5], dilations=[20, 10, 5])
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[3, 3, 3], paddings=[10, 5, 2], dilations=[10, 5, 2])
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[3], paddings=[20], dilations=[20])
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[3], paddings=[10], dilations=[10])
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[3], paddings=[5], dilations=[5])
            # x_pac_pool, _ = self.build_pac_pool_list(None, kernel_sizes=[7], paddings=[30], dilations=[10])
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[7], paddings=[15], dilations=[5])
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[7], paddings=[9], dilations=[3])
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[7], paddings=[3], dilations=[1])
            # x_pac_pool, _ = self.build_pac_pool_list(kernel_sizes=[9], paddings=[8], dilations=[2])

            x_pac_pool, _ = self.build_pac_pool_list('demo_layer', kernel_sizes=[15], strides=[15cd], paddings=[0], dilations=[1])

            im_in = input_extra_dict['im_trainval_RGB']
            # im_in = F.interpolate(im_in, [120, 160], mode='bilinear')
            im_in = F.interpolate(im_in, [60, 80], mode='bilinear')
            # im_in = F.interpolate(im_in, [30, 40], mode='bilinear')
            # x_pac_pool, _ = self.build_pac_pool_list(None, kernel_sizes=[kH], paddings=[padding], dilations=[dilation], if_deform=if_deform, native_impl=native_impl)

            # print(matseg_embeddings.shape, matseg_embeddings[0, :5, :2, 0])
            # matseg_embeddings = torch.ones_like(matseg_embeddings, device=matseg_embeddings.device)
            # im_trainval_RGB_mask_pooled_mean = im_in
            
            im_trainval_RGB_mask_pooled_mean, kernel_list = self.pac_pool_transform(im_in, (matseg_embeddings, mat_notlight_mask_gpu_float), x_pac_pool, force_mean=True, return_kernel_list=True, offsets=None)

            print(im_trainval_RGB_mask_pooled_mean.shape, '======')
        return_dict.update({'im_trainval_RGB_mask_pooled_mean': im_trainval_RGB_mask_pooled_mean, 'kernel_list': kernel_list})

        
        if 'x6' in self.albedo_pac_pool_mean_layers:
            x6 = self.pac_pool_transform(x6, (matseg_embeddings, mat_notlight_mask_gpu_float), self.x6_pac_pool)
        dx1 = F.relu(self.dgn1(self.dconv1(x6 ) ) )

        xin1 = torch.cat([dx1, x5], dim = 1)
        if 'xin1' in self.albedo_pac_pool_mean_layers:
            xin1 = self.pac_pool_transform(xin1, (matseg_embeddings, mat_notlight_mask_gpu_float), self.xin1_pac_pool)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear') ) ), True)

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear')
        xin2 = torch.cat([dx2, x4], dim=1 )
        if 'xin2' in self.albedo_pac_pool_mean_layers:
            xin2 = self.pac_pool_transform(xin2, (matseg_embeddings, mat_notlight_mask_gpu_float), self.xin2_pac_pool)
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear') ) ), True)

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear')
        xin3 = torch.cat([dx3, x3], dim=1)
        if 'xin3' in self.albedo_pac_pool_mean_layers:
            xin3 = self.pac_pool_transform(xin3, (matseg_embeddings, mat_notlight_mask_gpu_float), self.xin3_pac_pool)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear') ) ), True)

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear')
        xin4 = torch.cat([dx4, x2], dim=1 )
        if 'xin4' in self.albedo_pac_pool_mean_layers:
            xin4 = self.pac_pool_transform(xin4, (matseg_embeddings, mat_notlight_mask_gpu_float), self.xin4_pac_pool)
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear') ) ), True)

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear')
        xin5 = torch.cat([dx5, x1], dim=1 )
        if 'xin5' in self.albedo_pac_pool_mean_layers:
            xin5 = self.pac_pool_transform(xin5, (matseg_embeddings, mat_notlight_mask_gpu_float), self.xin5_pac_pool)
        dx6 = F.relu(self.dgn6(self.dconv6(F.interpolate(xin5, scale_factor=2, mode='bilinear') ) ), True)

        if dx6.size(3) != im.size(3) or dx6.size(2) != im.size(2):
            dx6 = F.interpolate(dx6, [im.size(2), im.size(3)], mode='bilinear')
        if 'xin6' in self.albedo_pac_pool_mean_layers:
            # print('>> xin6', dx6.shape)
            dx6 = self.pac_pool_transform(dx6, (matseg_embeddings, mat_notlight_mask_gpu_float), self.xin6_pac_pool)
            # print('>>>>> xin6', dx6.shape, self.get_in_c(64, 'xin6', self.xin6_pac_pool_len))
        x_orig = self.dconvFinal(self.dpadFinal(dx6 ) )

        # print(x1, x2, x3, x4, x5, x6)
        
        # print(x6.shape, dx1.shape, dx2.shape, dx3.shape, dx4.shape, dx5.shape, dx6.shape, x_orig.shape) 
        # torch.Size([16, 512, 7, 10]) torch.Size([16, 256, 15, 20]) torch.Size([16, 256, 30, 40]) torch.Size([16, 128, 60, 80]) torch.Size([16, 64, 120, 160]) torch.Size([16, 64, 240, 320]) torch.Size([16, 3, 240, 320])


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

        return_dict.update({'x_out': x_out})
        # if self.if_albedo_pooling:

        return return_dict
