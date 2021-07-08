'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''

import torch


def ray_from_gmm(
        Z_in, 
        mix_in, mu_in_, cov_in_, 
        imsize, nspixel, 
        abs_spixel_ind, 
        epsilon= 1e-9, 
        gamma=None, 
        gamma_rel=None,
        reduce_method = 'direct_weight_sum',
        correction_factor=1e8,
        return_inv_sigma= False,
        ):
    '''
    Take 1D slice of 3D GMMs

    input:
     Z is Nx3 (set of un-normalized rays)
     mix is Jx1
     mu is Jx3
     cov is Jx3x3
     imsize - (H, W)
     nspixel - int
     abs_spixel_ind_in - 9xHxW
     gamma - N x J
        
    optional:
     gamma_rel - N x 9
     reduce_method - {'softmax', 'direct_weight_sum'}

    output:
     mu_map, var_map - 9xHxW
    '''

    mu_in, cov_in = mu_in_, cov_in_ 
    indx_abs2rel = abs_spixel_ind.squeeze().reshape(9,-1).permute(1,0).contiguous() # 9xHxW -> Nx9 (N=HxW)

    H, W = imsize[0], imsize[1] 
    Z = Z_in
    mu = mu_in.T.unsqueeze(0) # 1x3xJ
    cov = cov_in.T # 3x3xJ
    rays_1d = torch.sqrt(torch.sum(Z*Z, axis=1)).T.unsqueeze(1) # Nx1, norm of un-nomailized rays
    ray_unit = Z/rays_1d # Nx3
    N = Z.shape[0]
    J = mu.shape[-1]

    # st = time.time()
    inv_cov = cov.T.inverse().T #3x3xJ NOTE: the inverse here may lead to NaN value!
    mix = mix_in.T #1xJ 

    _var1 = torch.einsum("ab,bcd->acd", (ray_unit, inv_cov)) # (Nx3)*(3x3xJ) ==> Nx3xJ
    _var5 = torch.einsum('ijk,ljk->ik',_var1, mu) # (Nx3xJ)*(1x3xJ) ==> NxJ  
    _var2 = _var1.T*ray_unit.T # (Jx3xN)*(3xN) ==> Jx3xN
    _var3 = torch.sum(_var2, axis=1).T # (Jx3xN) ==> JxN ==> NxJ

    var_1d_ = torch.clamp(1.0/_var3, min=epsilon) # NxJ: variance along each rays
    mu_1d_ = _var5* var_1d_ # NxJ 
    stddev_1d_ = torch.sqrt(var_1d_)

    # var_1d = torch.gather(var_1d_, dim=-1, index=indx_abs2rel) # NxJ -> Nx9
    mu_1d = torch.gather(mu_1d_, dim=-1, index=indx_abs2rel)
    stddev_1d = torch.gather(stddev_1d_, dim=-1, index=indx_abs2rel)

    if gamma_rel is None:
        assert gamma is not None
        gamma_rel= torch.gather(gamma, dim=-1, index=indx_abs2rel)

    mu_map = mu_1d.reshape(H, W, -1).permute( 2,0,1 ).contiguous()
    stddev_map = stddev_1d.reshape(H, W, -1).permute(2,0,1).contiguous()

    ratio = (Z[:,-1]/ (Z**2).sum(1).sqrt()).reshape(H, W).unsqueeze(0)
    mu_map = mu_map * ratio
    stddev_map = stddev_map * ratio

    gamma_rel_map=gamma_rel.T.contiguous().reshape(9, H, W)
    if reduce_method == 'direct_weight_sum':
        mu_weight_sum = (mu_map*gamma_rel_map).sum(dim=0) / ( gamma_rel_map.sum(dim=0) +1e-8 )
        sigma_weight_sum = (stddev_map*gamma_rel_map).sum(dim=0) / ( gamma_rel_map.sum(dim=0) +1e-8 )

    elif reduce_method == 'softmax':
        gamma_rel_map_softmax = (gamma_rel_map*correction_factor).softmax(dim=0)
        mu_weight_sum = (mu_map*gamma_rel_map_softmax).sum(dim=0) / ( gamma_rel_map_softmax.sum(dim=0) +1e-8)
        sigma_weight_sum = (stddev_map*gamma_rel_map_softmax).sum(dim=0) / ( gamma_rel_map_softmax.sum(dim=0) +1e-8 )

    else:
        raise NotImplementedError

    if return_inv_sigma:
        return mu_map, 1./ (stddev_map+1e-9), gamma_rel, mu_weight_sum, 1./ (sigma_weight_sum+1e-9)
    else:
        return mu_map, stddev_map, gamma_rel, mu_weight_sum, sigma_weight_sum 