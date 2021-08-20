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
import models_def.model_nvidia.ssn.ssn_fullJ as ssn_fullJ

class SSNFeatsTransformAdaptive(torch.nn.Module):
    '''
    feat transform with SSN and reconstruction
    '''

    def __init__(
            self, args, spixel_dims, n_iter=10, if_dense=False):
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

        self.if_dense = if_dense

    def forward(self, tensor_to_transform, feats_in=None, affinity_in=None, mask=None, scale_down_gamma_tensor=1, index_add=True, if_return_codebook_only=False):
        '''
        INPUTS:
        tensor_to_transform -      nbatch x D1 x H x W
        feats_in -      nbatch x D2 x H x W

        OUTPUT:
        abs_affinity -      nbatch x J x H x W
        tensor_recon -      nbatch x D1 x H x W
        '''

        if feats_in is None:
            feats_in = tensor_to_transform

        # If ture, the spixel index assignment for all pixels would be the inital grid initialization
        # ssn_grid_spixel = self.ssn_grid_spixel

        # feats_in = batch['feats_in']

        batch_size, D, H, W = feats_in.shape
        assert self.spixel_nums[0] <= H
        assert self.spixel_nums[1] <= W
        if mask is not None:
            assert mask.shape==(batch_size, H, W)
        J = self.spixel_nums[0] * self.spixel_nums[1]

        ssn.ssn_iter_to_use = ssn_fullJ.ssn_iter if self.if_dense else ssn.ssn_iter

        if affinity_in is None:
            abs_affinity, dist_matrix, spixel_features = \
                ssn.ssn_iter_to_use(
                    feats_in, n_iter=self.n_iter, 
                    num_spixels_width=self.num_spixels_width, 
                    num_spixels_height=self.num_spixels_height, 
                    mask=mask, 
                    index_add=index_add)
            abs_affinity = abs_affinity.view(batch_size, J, H, W)
        else:
            abs_affinity = affinity_in
        assert abs_affinity.shape[1]==J


        # print(abs_affinity.shape, dist_matrix.shape, spixel_features.shape) # torch.Size([1, J, H, W]) torch.Size([1, 9, HW]) torch.Size([1, D, J]); h, w being the dimensions of spixels, j=h*w
        

        recon_return_dict = self.recon_tensor(abs_affinity, tensor_to_transform, scale_down_gamma_tensor=scale_down_gamma_tensor, if_return_codebook_only=if_return_codebook_only)

        res = dict({
            'abs_affinity': abs_affinity, 
            'tensor_recon': recon_return_dict['I_hat'], 
            'Q': recon_return_dict['Q'], 
            'C': recon_return_dict['C'], 
            'Q_2D': recon_return_dict['Q_2D'], 
        })


        return  res

    def recon_tensor(self, gamma, tensor_to_transform, scale_down_gamma_tensor=1, if_return_codebook_only=False):
        '''
        gamma: Q, BxJxHxW
        tensor_to_transform: BxDxHxW, where N*scale_down_gamma_tensor=HW
        scale_down_gamma_tensor
        '''
        if type(scale_down_gamma_tensor) is tuple:
            scale_down_gamma, scale_down_tensor = scale_down_gamma_tensor[0], scale_down_gamma_tensor[1]
        else:
            scale_down_gamma = scale_down_gamma_tensor
            scale_down_tensor = 1
            
        if scale_down_gamma != 1: # scale Q
            # print(gamma.shape, tensor_to_transform.shape, scale_down_gamma_tensor)
            gamma_resized = F.interpolate(gamma, scale_factor=1./float(scale_down_gamma), mode='bilinear')
            gamma_resized = gamma_resized / (torch.sum(gamma_resized, 1, keepdims=True)+1e-6)
            gamma = gamma_resized
        if scale_down_tensor != 1: # scale tensor
            # print(gamma.shape, tensor_to_transform.shape, scale_down_gamma_tensor)
            tensor_to_transform_resized = F.interpolate(tensor_to_transform, scale_factor=1./float(scale_down_tensor), mode='bilinear')
            tensor_to_transform_resized = tensor_to_transform_resized / (torch.sum(tensor_to_transform_resized, 1, keepdims=True)+1e-6)
            tensor_to_transform = tensor_to_transform_resized
        
        if gamma.shape[-2::]!=tensor_to_transform.shape[-2::]:
            print('gamma.shape[-2::]!=tensor_to_transform.shape[-2::]!', gamma.shape, tensor_to_transform.shape)
            assert False

        batch_size, J = gamma.shape[0], gamma.shape[1]
        batch_size_, D, H, W = tensor_to_transform.shape
        N = H * W
        assert batch_size_==batch_size

        Q_M_Jnormalized = gamma / (gamma.sum(-1, keepdims=True).sum(-2, keepdims=True)+1e-6) # [B, J, H, W]
        tensor_to_transform_flattened = tensor_to_transform.permute(0, 2, 3, 1).view(batch_size, -1, D)
        tensor_to_transform_J = Q_M_Jnormalized.view(batch_size, J, -1) @ tensor_to_transform_flattened # [B, J, D], the code book
        tensor_to_transform_J = tensor_to_transform_J.permute(0, 2, 1) # [B, D, J], the code book

        if if_return_codebook_only:
            im_single_hat = None
        else:
            im_single_hat = tensor_to_transform_J @ gamma.view(batch_size, J, N) # (B, D, N) where N = H * W
            im_single_hat = im_single_hat.view(batch_size, D, H, W)

        return {'C': tensor_to_transform_J, 'Q': gamma.view(batch_size, J, N), 'I_hat': im_single_hat, 'Q_2D': gamma}