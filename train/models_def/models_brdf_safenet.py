import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import pac
from models_def.model_matseg import logit_embedding_to_instance
import torch.nn.functional as F

class decoder0_safenet(nn.Module):
    def __init__(self, opt, mode=-1, out_channel=3, input_dict_guide=None):
        super(decoder0_safenet, self).__init__()
        self.mode = mode
        self.opt = opt

        self.albedo_safenet_affinity_layers = self.opt.cfg.MODEL_MATSEG.albedo_safenet_affinity_layers

        self.conv_layers_num = {name: None for name in self.opt.cfg.MODEL_MATSEG.albedo_safenet_affinity_layers_allowed}

        self.safenet_module_dict = torch.nn.ModuleDict([])

        if 'x6' in self.albedo_safenet_affinity_layers:
            self.build_affinity_prop('x6', 1024)
        self.dconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding = 1, bias=True) #in: [16, 512, 7, 10]
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512 )

        if 'xin1' in self.albedo_safenet_affinity_layers:
            self.build_affinity_prop('xin1', 1024)
        self.dconv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding = 1, bias=True) #in: [16, 512, 7, 10]
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        if 'xin2' in self.albedo_safenet_affinity_layers:
            self.build_affinity_prop('xin2', 512)
        self.dconv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True) #in: [16, 256, 15, 20]
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        if 'xin3' in self.albedo_safenet_affinity_layers:
            self.build_affinity_prop('xin3', 512)
        self.dconv4 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True) #in: [16, 256, 30, 40]
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=128 )

        if 'xin4' in self.albedo_safenet_affinity_layers:
            self.build_affinity_prop('xin4', 256)
        self.dconv5 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True) # in: [16, 128, 60, 80]
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=64 )

        if 'xin5' in self.albedo_safenet_affinity_layers:
            self.build_affinity_prop('xin5', 128)
        self.dconv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True) # in: [16, 64, 120, 160]
        self.dgn6 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dpadFinal = nn.ReplicationPad2d(1)
        if 'xin6' in self.albedo_safenet_affinity_layers:
            self.build_affinity_prop('xin6', 64)
        self.dconvFinal = nn.Conv2d(in_channels=64, out_channels=3, kernel_size = 3, stride=1, bias=True) # in: [16, 64, 240, 320]

        self.flag = True

    def build_affinity_prop(self, layer_name, in_channels):
        self.safenet_module_dict['%s_1x1_1'%layer_name] = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.safenet_module_dict['%s_1x1_2'%layer_name]  = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.safenet_module_dict['%s_bn'%layer_name]  = nn.BatchNorm2d(in_channels)

    def affinity_prop(self, layer_name, xin, matseg_embeddings, mat_notlight_mask_gpu_float):
        xin_conv1 = self.safenet_module_dict['%s_1x1_1'%layer_name](xin)
        N, C, H, W = xin_conv1.shape
        xin_conv1 = xin_conv1.permute(0, 2, 3, 1).view(N, H*W, C)
        embeddings = self.process_embedding(xin, matseg_embeddings, mat_notlight_mask_gpu_float)
        assert embeddings.shape == (N, self.opt.cfg.MODEL_MATSEG.matseg_embed_dims, H, W)
        if self.opt.cfg.MODEL_MATSEG.if_albedo_safenet_normalize_embedding:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings_2 = embeddings.view(N, self.opt.cfg.MODEL_MATSEG.matseg_embed_dims, H*W)

        if self.opt.cfg.MODEL_MATSEG.if_albedo_safenet_use_pacnet_affinity:
            embed_expand = embeddings_2.unsqueeze(-1).expand(-1, -1, -1, H*W) # [N, D, H*W, H*W]
            # print(embed_expand.shape, embed_expand.transpose(2, 3).shape)
            A = torch.exp(-0.5 * (torch.norm(embed_expand - embed_expand.transpose(2, 3), dim=1)**2))
        else:
            embeddings_1 = embeddings.permute(0, 2, 3, 1).view(N, H*W, self.opt.cfg.MODEL_MATSEG.matseg_embed_dims)
            A = torch.exp(torch.matmul(embeddings_1, embeddings_2))

        A = A / (A.sum(dim=2, keepdims=True)) # normalize row-wise
        # a1 = a / a.sum(dim=1, keepdims=True)
        # A = torch.transpose(F.normalize(A, p=1, dim=1), 1, 2)
        xin_conv1_transformed = torch.matmul(A, xin_conv1)
        assert xin_conv1_transformed.shape == (N, H*W, C)
        xin_conv1_transformed = xin_conv1_transformed.transpose(1, 2).view(N, C, H, W)
        xin_conv1_transformed_2 = self.safenet_module_dict['%s_1x1_2'%layer_name](xin_conv1_transformed)
        xin = xin + self.safenet_module_dict['%s_bn'%layer_name](xin_conv1_transformed_2)
        return xin

    def process_embedding(self, x, matseg_embeddings, mat_notlight_mask_gpu_float):
        if matseg_embeddings.size(3) != x.size(3) or matseg_embeddings.size(2) != x.size(2):
            matseg_embeddings_use = F.interpolate(matseg_embeddings, [x.size(2), x.size(3)], mode='bilinear')
            mat_notlight_mask_gpu_float_use = F.interpolate(mat_notlight_mask_gpu_float, [x.size(2), x.size(3)], mode='nearest')
        else:
            matseg_embeddings_use = matseg_embeddings
            mat_notlight_mask_gpu_float_use = mat_notlight_mask_gpu_float
        matseg_embeddings_use = matseg_embeddings_use * (2. * mat_notlight_mask_gpu_float_use - 1)
        return matseg_embeddings_use

    def forward(self, im, x1, x2, x3, x4, x5, x6, input_dict_extra=None):

        return_dict = {}

        matseg_embeddings = input_dict_extra['matseg-embeddings']
        # matseg_embeddings = matseg_embeddings * (2. * input_dict_extra['mat_notlight_mask_gpu_float'] - 1)
        mat_notlight_mask_gpu_float = input_dict_extra['mat_notlight_mask_gpu_float']

        im_in_transformed, kernel_list, A, sample_ij, embeddings = None, None, None, None, None
        
        # assert self.opt.cfg.MODEL_MATSEG.albedo_pooling_debug == False
        if self.opt.cfg.MODEL_MATSEG.albedo_pooling_debug and self.opt.if_vis_debug_pac:
            im_in = input_dict_extra['im_trainval_RGB']
            im_in = F.interpolate(im_in, [120, 160], mode='bilinear')
            # im_in = F.interpolate(im_in, [60, 80], mode='bilinear')
            # im_in = F.interpolate(im_in, [30, 40], mode='bilinear')
            N, C, H, W = im_in.shape
            embeddings = self.process_embedding(im_in, matseg_embeddings, mat_notlight_mask_gpu_float)
            assert embeddings.shape == (N, self.opt.cfg.MODEL_MATSEG.matseg_embed_dims, H, W)
            
            # embeddings = F.normalize(embeddings, p=2, dim=1)

            embeddings_2 = embeddings.contiguous().view(N, self.opt.cfg.MODEL_MATSEG.matseg_embed_dims, -1)

            # SAFENET
            embeddings_1 = embeddings.permute(0, 2, 3, 1).view(N, H*W, self.opt.cfg.MODEL_MATSEG.matseg_embed_dims)
            A = torch.exp(torch.matmul(embeddings_1, embeddings_2))
            # A = torch.sum(torch.exp(-0.5 * embeddings_1.unsqueeze(2).expand(-1, -1, H*W, -1) * embeddings_1.unsqueeze(1).expand(-1, H*W, -1, -1)), -1)
            print(A.shape, '====-------')
            # A = torch.transpose(A / A.sum(dim=1, keepdims=True), 1, 2)
            # A = torch.transpose(F.normalize(A, p=1, dim=2), 1, 2)
            A = A / (A.sum(dim=2, keepdims=True))
            # A = F.normalize(A, p=1, dim=0)
            print(torch.sum(A[0], dim=1))

            # Pac-conv
            # embed_expand = embeddings_2.unsqueeze(-1).expand(-1, -1, -1, H*W) # [N, D, H*W, H*W]
            # print(embed_expand.shape, embed_expand.transpose(2, 3).shape)
            # A = torch.exp(-0.5 * (torch.norm(embed_expand - embed_expand.transpose(2, 3), dim=1)**2))
            # A = A / (A.sum(dim=2, keepdims=True)) # [N, D, H*W, H*W]

            ## Visualize transformed image
            im_in = im_in.permute(0, 2, 3, 1).view(N, H*W, C)
            im_in_transformed = torch.matmul(A, im_in)
            assert im_in_transformed.shape == (N, H*W, C)
            im_in_transformed = im_in_transformed.transpose(1, 2).view(N, C, H, W)
            print(im_in_transformed.shape, '======')
            
            # Visualize sampled affinity matrix
            # sample_ij = [[1, 1], [1, 100], [1, 200], [101, 1], [101, 100], [101, 200], [201, 1], [201, 100], [201, 200]]
            # i_s = np.linspace(0, 220, 12) / 2.
            # j_s = np.linspace(0, 300, 16) / 2.
            # ij_s = np.meshgrid(i_s, j_s)
            # sample_ij = [[i, j] for i, j in zip(ij_s[0].flatten().astype(np.int16).tolist(), ij_s[1].flatten().astype(np.int16).tolist())]
            # embeddings_sampled = torch.cat([embeddings[:, :, i:i+1, j] for i, j in sample_ij], 2).permute(0, 2, 1) # [N, n, embed_dims]
            # A = torch.exp(torch.matmul(embeddings_sampled, embeddings_2))
            # A = A / (A.sum(dim=2, keepdims=True))

        return_dict.update({'im_trainval_RGB_mask_pooled_mean': im_in_transformed, 'kernel_list': kernel_list, 'affinity': A, 'sample_ij': sample_ij, 'embeddings': embeddings})

        
        if 'x6' in self.albedo_safenet_affinity_layers:
            self.affinity_prop('x6', x6, matseg_embeddings, mat_notlight_mask_gpu_float)

        dx1 = F.relu(self.dgn1(self.dconv1(x6 ) ) )

        xin1 = torch.cat([dx1, x5], dim = 1)
        if 'xin1' in self.albedo_safenet_affinity_layers:
            self.affinity_prop('xin1', xin1, matseg_embeddings, mat_notlight_mask_gpu_float)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear') ) ), True)

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear')
        xin2 = torch.cat([dx2, x4], dim=1 )
        if 'xin2' in self.albedo_safenet_affinity_layers:
            self.affinity_prop('xin2', xin2, matseg_embeddings, mat_notlight_mask_gpu_float)
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear') ) ), True)

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear')
        xin3 = torch.cat([dx3, x3], dim=1)
        if 'xin3' in self.albedo_safenet_affinity_layers:
            self.affinity_prop('xin3', xin3, matseg_embeddings, mat_notlight_mask_gpu_float)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear') ) ), True)

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear')
        xin4 = torch.cat([dx4, x2], dim=1 )
        if 'xin4' in self.albedo_safenet_affinity_layers:
            self.affinity_prop('xin4', xin4, matseg_embeddings, mat_notlight_mask_gpu_float)
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear') ) ), True)

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear')
        xin5 = torch.cat([dx5, x1], dim=1 )
        if 'xin5' in self.albedo_safenet_affinity_layers:
            self.affinity_prop('xin5', xin5, matseg_embeddings, mat_notlight_mask_gpu_float)
        dx6 = F.relu(self.dgn6(self.dconv6(F.interpolate(xin5, scale_factor=2, mode='bilinear') ) ), True)

        if dx6.size(3) != im.size(3) or dx6.size(2) != im.size(2):
            dx6 = F.interpolate(dx6, [im.size(2), im.size(3)], mode='bilinear')
        if 'xin6' in self.albedo_safenet_affinity_layers:
            self.affinity_prop('xin6', xin6, matseg_embeddings, mat_notlight_mask_gpu_float)
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

