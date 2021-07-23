'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''

import torch
from torch.functional import Tensor
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import models_def.model_nvidia.ssn.ssn as ssn

class SSNFeatsTransformAdaptive(torch.nn.Module):
    '''
    feat transform with SSN and reconstruction
    '''

    def __init__(
            self, args, spixel_dims, n_iter=10):
        '''
        '''
        super().__init__()
        # self.learning_rate = args.cfg.MODEL_GMM.learning_rate
        # self.ssn_grid_spixel= args.cfg.MODEL_GMM.ssn_grid_spixel
        self.ssn_grid_spixel = False
        # self.src_idx= args.cfg.MODEL_GMM.src_idx

        # self.spixel_nums =  (21, 15)  #w, h
        # self.spixel_nums =  (32, 24)  #w, h
        # self.spixel_nums =  (16, 12)  #w, h
        # self.spixel_nums =  (8, 6)  #w, h
        self.spixel_nums = spixel_dims
        self.num_spixels_height, self.num_spixels_width = self.spixel_nums[0], self.spixel_nums[1]
        self.num_spixels = self.spixel_nums[0]*self.spixel_nums[1] 
        
        self.n_iter = n_iter

        self.total_step=0

        self.opt = args

    def forward(self, feats_in, tensor_to_transform=None, index_add=True):
        '''
        INPUTS:
        feats_in -      nbatch x D x H x W
        '''

        if tensor_to_transform is None:
            tensor_to_transform = feats_in

        # If ture, the spixel index assignment for all pixels would be the inital grid initialization
        # ssn_grid_spixel = self.ssn_grid_spixel

        # feats_in = batch['feats_in']

        J = self.spixel_nums[0] * self.spixel_nums[1]
        batch_size, D, H, W = feats_in.shape
        assert self.spixel_nums[0] <= H
        assert self.spixel_nums[1] <= W


        # for sample_idx in range(batch_size):
        #     #for demo, use gt depth, and gt poses
        #     feats_in_single = feats_in[sample_idx:sample_idx+1]

        #     # abs_affinity, dist_matrix, spixel_features = align.feat2gmm(feats_in_single, ssn_iter=10)
        #     # spixel_dim = feats_in_single.shape[2:4]
        #     # num_spixels_height, num_spixels_width = spixel_dim[1], spixel_dim[0] 
        #     # num_spixels_height, num_spixels_width = self.spixel_nums
        #     # num_spixels = num_spixels_height * num_spixels_width

        #     # print(feats_in_single.shape) # torch.Size([1, D, H, W])

        #     abs_affinity, dist_matrix, spixel_features = \
        #         ssn.ssn_iter(
        #             feats_in_single, n_iter=10, 
        #             num_spixels_width=self.num_spixels_width, 
        #             num_spixels_height=self.num_spixels_height)


        #     # print(abs_affinity.shape, dist_matrix.shape, spixel_features.shape) # torch.Size([1, J, H, W]) torch.Size([1, 9, HW]) torch.Size([1, D, J]); h, w being the dimensions of spixels, j=h*w
            
        #     abs_affinity = abs_affinity.view(1, J, H, W)
        #     abs_affinity_list.append(abs_affinity)
            
        #     # spixel_features = spixel_features.view(feats_in_single.shape[0], self.num_spixels, *feats_in_single.shape[-2:]).contiguous()

        #     feats_recon = self.recon_feats(abs_affinity, feats_in_single)
        #     feats_recon_list.append(feats_recon)


        #     torch.cuda.empty_cache()

        # res = dict({
        #     'abs_affinity': torch.cat(abs_affinity_list, 0), 
        #     'feats_recon': torch.cat(feats_recon_list, 0)
        # })

        abs_affinity, dist_matrix, spixel_features = \
            ssn.ssn_iter(
                feats_in, n_iter=self.n_iter, 
                num_spixels_width=self.num_spixels_width, 
                num_spixels_height=self.num_spixels_height, 
                index_add=index_add)


        # print(abs_affinity.shape, dist_matrix.shape, spixel_features.shape) # torch.Size([1, J, H, W]) torch.Size([1, 9, HW]) torch.Size([1, D, J]); h, w being the dimensions of spixels, j=h*w
        
        abs_affinity = abs_affinity.view(batch_size, J, H, W)

        feats_recon = self.recon_feats(abs_affinity, tensor_to_transform)

        res = dict({
            'abs_affinity': abs_affinity, 
            'feats_recon': feats_recon
        })


        return  res

    def recon_feats(self, gamma, feat_map, scale_feat_map=1):
        '''
        gamma: Q, BxJxHxW
        feat_map: BxDxHxW, where N*scale_feat_map=HW
        scale_feat_map
        '''
        if scale_feat_map != 1:
            # print(gamma.shape, feat_map.shape, scale_feat_map)
            gamma_resized = F.interpolate(gamma, scale_factor=1./float(scale_feat_map))
            gamma_resized = gamma_resized / (torch.sum(gamma_resized, 1, keepdims=True)+1e-6)
            gamma = gamma_resized
            
        assert gamma.shape[-2::]==feat_map.shape[-2::]

        batch_size, J = gamma.shape[0], gamma.shape[1]
        batch_size_, D, H, W = feat_map.shape
        N = H * W
        assert batch_size_==batch_size

        Q_M_Jnormalized = gamma / (gamma.sum(-1, keepdims=True).sum(-2, keepdims=True)+1e-6) # [B, J, 240, 320]
        feat_map_flattened = feat_map.permute(0, 2, 3, 1).view(batch_size, -1, D)
        feat_map_J = Q_M_Jnormalized.view(batch_size, J, -1) @ feat_map_flattened # [b, J, D], the code book
        feat_map_J = feat_map_J.permute(0, 2, 1) # [b, D, J], the code book
        # print(feat_map_J.shape, gamma.view(batch_size, J, N).shape)
        im_single_hat = feat_map_J @ gamma.view(batch_size, J, N) # (b, D, N)
        im_single_hat = im_single_hat.view(batch_size, D, H, W)

        return im_single_hat