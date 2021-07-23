'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''

import math
import torch
import os, glob

from .pair_wise_dist_gmm import PairwiseDistFunction
from models_def.model_nvidia.utils_nvidia.misc import outer_prod, outer_prod_batch

import torch.cuda.amp as amp

# for debugging
# from matplotlib.pyplot import * 
import time
import torch.nn.functional as F

@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    # import ipdb; ipdb.set_trace()
    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)

def affinity_rel2abs(affinity_matrix, abs_indices, num_spixels, hard_labels=None):
    '''
    get abs_affinity matrix
    
    INPUTS:

    optional 
      hard_labels - 9 H W

     OUTPUTS
        B J N
    '''
    # print(affinity_matrix.shape, abs_indices.shape) # torch.Size([1, 9, 98304]) torch.Size([3, 884736])

    reshaped_affinity_matrix = affinity_matrix.reshape(-1) 
    mask = (abs_indices[1] >= 0)*(abs_indices[1] < num_spixels)
    # print(abs_indices.shape, mask.shape) # torch.Size([3, 884736]) torch.Size([884736])

    # abs_indices: 3x(9HW), 
    # affinity_matrix: Bx9x(HW) B=1, 
    # reshaped_affinity_matrix: (9BHW), B=1
    # print(mask.shape, abs_indices.shape, reshaped_affinity_matrix.shape)
    sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])
    # print(abs_indices[:, mask].shape, reshaped_affinity_matrix[mask].shape, sparse_abs_affinity.shape) # torch.Size([3, 843786]) torch.Size([843786]) torch.Size([1, 315, 98304])
    abs_affinity = sparse_abs_affinity.to_dense().contiguous()  #BxJx(HW), B=1

    return abs_affinity 

def cal_det_sym3x3(cov, return_inv=False, inv_eps = 0.):
    '''
    cov - * x 3 x 3
    return_inv - if return inverse
    Ref: 
    https://math.stackexchange.com/questions/603213/simplified-method-for-symmetric-matrix-determinants
    https://math.stackexchange.com/questions/233378/inverse-of-a-3-x-3-covariance-matrix-or-any-positive-definite-pd-matrix
    return - * vector
    '''
    a = cov[:,0,0]
    b = cov[:,0,1]
    c = cov[:,0,2]
    d = cov[:,1,1]
    e = cov[:,1,2]
    f = cov[:,2,2]

    cov_det = a*(d*f-e**2) + b*(c*e - b*f) + c*(b*e - d*c)

    if return_inv:
        cov_inv = torch.zeros(cov.shape, device=cov.device)
        cov_inv[:,0,0]= f * d - e**2  
        cov_inv[:,0,1] = c * e - f * b  
        cov_inv[:,0,2] = b * e - c * d  

        cov_inv[:,1,0] = cov_inv[:, 0,1]
        cov_inv[:,1,1] = f * a - c**2  
        cov_inv[:,1,2] = b * c - a * e 

        cov_inv[:,2,0]= cov_inv[:, 0,2]
        cov_inv[:,2,1]= cov_inv[:, 1,2]
        cov_inv[:,2,2]= a * d - b**2
        cov_inv = cov_inv / (cov_det.unsqueeze(-1).unsqueeze(-1) + inv_eps)

    if not return_inv:
        return cov_det
    else:
        return cov_det, cov_inv

def get_det_robust(spixel_cov, regularizer=1e-3):
    '''
    spixel_cov - * x 3 x 3
    '''
    trace = spixel_cov[:, 0,0] + spixel_cov[:, 1,1] + spixel_cov[:, 2,2]
    trace = trace.unsqueeze(-1).unsqueeze(-1) /3.0 # *x1x 1
    eyes = torch.eye(3, device=spixel_cov.device).unsqueeze(0) # 1x3x3
    spixel_cov_reg = (1-regularizer)*spixel_cov + regularizer*eyes*trace
    return spixel_cov_reg.det() 

def regularzie_cov(spixel_cov, regularizer=1e-3, reg2 = 0.):
    '''
    cov - * x 3 x 3
    '''
    trace = spixel_cov[:, 0,0] + spixel_cov[:, 1,1] + spixel_cov[:, 2,2]
    trace = trace.unsqueeze(-1).unsqueeze(-1) /3.0 # *x1x 1

    #
    eyes = torch.eye(3, device=spixel_cov.device).unsqueeze(0) # 1x3x3
    cov_reg = (1-regularizer)*spixel_cov + regularizer*eyes*trace 

    #NOTE 1: if we add this regularizer (reg2=1e-2), then the resampled depth would be much blurry
    #NOTE 2: if we don't add it, then there could be zero matrix in cov_reg
    if reg2>0:
        # print('regulariz_cov: add reg2')
        cov_reg += reg2 * eyes 

    return cov_reg 

def M_step0(Gamma, pts, pts_outer, ):
    '''
    input: 
    Gamma - JxN
    pts - Nx3
    pts_outer - Nx9
    output:
    mu - Jx3
    sigma - Jx9
    mix - Jx1
    '''
    assert Gamma.ndim==2 and pts.ndim==2 and pts_outer.ndim==2, 'inputs should be 2-D arrays'
    T0 = Gamma.sum(1).unsqueeze(1) #Jx1
    T1 = torch.matmul(Gamma, pts) # Jx3
    T2 = torch.matmul(Gamma, pts_outer) #Jx9

    mix = T0/pts.shape[0] #Jx1
    mu = T1/T0 #Jx3
    #Get outer mu
    outer_mu = outer_prod(mu, mu).reshape(-1, 9)
    #Get sigma
    sigma = T2/T0 - outer_mu #Jx9

    # sigma_M = sigma.reshape(-1,3,3)
    # import ipdb; ipdb.set_trace()

    return mu, sigma, mix

def M_step(Gamma_in, pts, pts_outer, weights=None, robust_divide=False, epsilon = 1e-5):
    '''
    input: 
    Gamma - JxN
    pts - Nx3
    pts_outer - Nx9

    weights - 1XN

    output:
    mu - Jx3
    sigma - Jx9
    mix - Jx1
    '''


    if weights is not None:
        Gamma = Gamma_in * weights
    else:
        Gamma = Gamma_in

    assert Gamma.ndim==2 and pts.ndim==2 and pts_outer.ndim==2, 'inputs should be 2-D arrays'
    assert Gamma.shape[-1] == pts.shape[0] == pts_outer.shape[0] 

    T0 = Gamma.sum(1).unsqueeze(1) #Jx1
    T1 = torch.matmul(Gamma, pts) # Jx3
    T2 = torch.matmul(Gamma, pts_outer) #Jx9

    if weights is not None:
        mix = T0/ weights.sum() #Jx1 
    else:
        mix = T0/pts.shape[0] #Jx1

    if robust_divide:
        mu = T1/ torch.clamp(T0, min=epsilon) #Jx3
        #Get outer mu
        outer_mu = outer_prod(mu, mu).reshape(-1, 9)
        #Get sigma
        sigma = T2/ torch.clamp(T0, min=epsilon) - outer_mu #Jx9
    else:
        mu = T1/ T0
        #Get outer mu
        outer_mu = outer_prod(mu, mu).reshape(-1, 9)
        #Get sigma
        sigma = T2/ T0 - outer_mu #Jx9

    return mu, sigma, mix
    
def get_init_params(images, Z_outer, num_spixels_width, num_spixels_height):
    """
    calculate initial superpixels 
    
    Args:
        images: torch.Tensor
            A Tensor of shape (B, 3, H, W)
        Z_outer: 
            Nx9, each row outer prodcut of z_i : matmul(z_i, z_i.T)
        num_spixels_width: int
            A number of superpixels in each column
        num_spixels_height: int
            A number of superpixels int each row
    Return:
        mu: torch.Tensor
        Jx3
        sigma: torch.Tensor
        Jx9
        mix: torch.Tensor
        Jx1
        init_label_map: torch.Tensor
            A Tensor of shape (B, H * W)
        abs_indices: used in affinity_rel2abs()
    """
    # // pixel_mu - Bx3xN_pix
    # // spixel_mu - Bx3xN_spix
    # // spixel_invcov - Bx9xN_spix
    # // cmix - Bx1xN_spix. The log of the normalization constant multiplied with the mix parameter
    # // spixel_indices - Bx1xN_pix
    batchsize, channels, height, width = images.shape
    assert batchsize==1, 'only implemented for 1 batch!'
    device = images.device
    centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))
    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)
        init_label_map = init_label_map.reshape(batchsize, -1)
        abs_indices = get_abs_indices(init_label_map, num_spixels_width) 
        affinity_m_init = torch.ones(batchsize, 9, height*width, device=device) * 1./9.

    centroids = centroids.reshape(batchsize, channels, -1) #1x3xJ for reference, compare tomu

    abs_affinity = affinity_rel2abs(affinity_m_init, abs_indices, num_spixels).squeeze() #JxN_pix, \Gamma matrix
    abs_affinity = abs_affinity / abs_affinity.sum(0).unsqueeze(0) # normalize 

    T0 = abs_affinity.sum(1).unsqueeze(1) #Jx1
    T1 = torch.matmul(abs_affinity, images.squeeze().permute(1,2,0).reshape(-1,3)) # Jx3
    # print(abs_affinity.shape, T0.shape, images.shape, T1.shape, Z_outer.shape) # torch.Size([12, 315, 76800]) torch.Size([12, 315, 1]) torch.Size([12, 3, 240, 320])
    T2 = torch.matmul(abs_affinity, Z_outer) #Jx9
    mix = T0/ (height*width) #Jx1
    mu = T1/ T0 #Jx3
    #Get outer mu
    outer_mu = outer_prod(mu, mu).reshape(-1, 9)
    # print(outer_mu.shape, T2.shape, T0.shape) # torch.Size([315, 9]) torch.Size([315, 9]) torch.Size([315, 1])
    #Get sigma
    sigma = T2/T0 - outer_mu #Jx9

    return mu, sigma, mix, init_label_map, abs_indices

def ssn3d_iter(
        pixel_features, 
        n_iter, num_spixels_width, num_spixels_height, 
        if_timing=False, 
        return_abs_ind=False,
        reg = 1e-2, reg2=0, 
        Weights= None, 
        return_affinity_mat=False,
        bound_hard_labels=False,
        use_indx_init=False):
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6

    Weights - 1x N 

    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        n_iter: int
            A number of iterations
    """
    #TODO: ssn3d_iter suffers from numerical issues for mixed precision

    assert pixel_features.shape[1] ==3, 'ssn3d_iter() only deals with point cloud!'
    height, width = pixel_features.shape[-2:]
    num_spixels = num_spixels_width * num_spixels_height

    # get initial parameters for GMMs (mu, sigma, pi)
    permuted_pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1).permute(0, 2, 1).contiguous() #Bx N_pix x C
    pts_outer = outer_prod(permuted_pixel_features.squeeze(), permuted_pixel_features.squeeze()).reshape(-1, 9)
    # print(permuted_pixel_features.shape, pts_outer.shape)
    # print(spixel_mu.shape, spixel_cov.shape, mix.shape, init_label_map.shape, abs_indices.shape)

    # get sigma_invcov, cmix
    det_spixel_cov = spixel_cov.reshape(-1, 3, 3).det().unsqueeze(1) #Jx1 #TODO: will the explicit computation for det be faster than .det()? 
    print()
    #heuristic to takle negative det_spixel_cov
    det_spixel_cov = det_spixel_cov.abs()
    cmix = torch.log(mix+1e-10) - 1.5*torch.log(torch.FloatTensor([2*math.pi]).to(mix)) - .5*torch.log(det_spixel_cov + 1e-10)  #TODO: set 1.5*log(..) as a constant

    #less robust 3x3 inversion
    # spixel_invcov = spixel_cov.reshape(-1, 3, 3).inverse().reshape(1, -1, 9).permute(0, 2, 1).contiguous()

    #more robust 3x3 inversion
    det_spixel_cov_, spixel_invcov_ = cal_det_sym3x3(spixel_cov.reshape(-1,3,3) , return_inv=True, inv_eps=1e-10)
    spixel_invcov_[det_spixel_cov_==0,...] += 1e10 * torch.eye(3, device=spixel_invcov_.device).unsqueeze(0)
    spixel_invcov = spixel_invcov_.reshape(1, -1, 9).permute(0,2,1).contiguous()

    #prepare inputs
    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1) #BxCxN_pix
    spixel_mu = spixel_mu.T.unsqueeze(0)
    cmix = cmix.T.unsqueeze(0)
    

    if if_timing:
        time_E, time_rel2abs, time_M, time_regu, time_det, time_inv = 0, 0, 0, 0, 0, 0
        st = time.time()

    # BEGIN ITERATION #
    for i_iter in range(n_iter):

        # E-step: calculate /gammas
        # // pixel_mu - Bx3xN_pix
        # // spixel_mu - Bx3xN_spix
        # // spixel_invcov - Bx9xN_spix
        # // cmix - Bx1xN_spix. The log of the normalization constant multiplied with the mix parameter
        # // spixel_indices - Bx1xN_pix
        if if_timing: st_Estep = time.time()
        dist_matrix = PairwiseDistFunction.apply(
            pixel_features.contiguous(), spixel_mu.contiguous(), spixel_invcov.contiguous(), 
            cmix.contiguous(), init_label_map,
            num_spixels_width, num_spixels_height) # Bx9xN_pix, log(mixture * normalization_const * exp(-.5*maha_distance) ) = cmix - .5*maha_distance 
        if if_timing: time_E += time.time() - st_Estep

        if if_timing: st_rel2abs = time.time()
        affinity_matrix = (dist_matrix).softmax(1) # Bx9xN_pix

        #--debug:
        if affinity_matrix.isnan().sum()>0:
            print('affinity_matrix nan value happens')
            import ipdb; ipdb.set_trace()
        #debug end -- 
        
        abs_affinity = affinity_rel2abs(affinity_matrix, abs_indices, num_spixels) # B x J x N_pix
        # import ipdb; ipdb.set_trace()
        if if_timing: time_rel2abs += time.time() - st_rel2abs

        # M-step: update the centroids
        if if_timing: st_M = time.time()

        if Weights is None:
            spixel_mu, spixel_cov_, mix = M_step(abs_affinity.squeeze(0), pixel_features.squeeze(0).permute(1,0), pts_outer, robust_divide=True, epsilon=1e-10)
        else:
            spixel_mu, spixel_cov_, mix = M_step(abs_affinity.squeeze(0), pixel_features.squeeze(0).permute(1,0), pts_outer, weights= Weights, robust_divide=True, epsilon=1e-10)

        if if_timing: time_M += time.time() - st_M

        #|NOTE: Here we have to regularzie the cov matrix. Because some
        #|convariance matrace in spixel_cov_ are very flat (e.g. for depth on
        #|the edges). For those matraces the det is very close to zero or even
        #|negative. As they become negative, their inverse matrix's det would
        #|be negative with huge abs value This leads to a very huge value
        #|before we do softmax. If we do softmax directly on those clusters,
        #|all pixels will be assigned with very high weights to those clusters
        if if_timing:
            st_regu = time.time()

        spixel_cov = regularzie_cov(\
                spixel_cov_.reshape(-1,3,3), 
                regularizer= reg, reg2= reg2).reshape(spixel_cov_.shape)

        if if_timing:
            time_regu += time.time() -st_regu

        if if_timing:
            st_det = time.time()
        det_spixel_cov_, spixel_invcov_ = cal_det_sym3x3(spixel_cov.reshape(-1,3,3) , return_inv=True, inv_eps=1e-10)

        #regularzie spixe_inv_cov_ for zero spixel_cov#
        spixel_invcov_[det_spixel_cov_==0,...] += 1e10 * torch.eye(3, device=spixel_invcov_.device).unsqueeze(0)

        det_spixel_cov = det_spixel_cov_.unsqueeze(1).abs()

        if if_timing:
            time_det += time.time()- st_det

        if if_timing:
            st_inv = time.time()

        cmix = torch.log(mix + 1e-10) \
                - 1.5*torch.log(torch.FloatTensor([2*math.pi]).to(mix)) \
                - .5*torch.log(det_spixel_cov + 1e-10)  #TODO: set 1.5*log(2pi) as a constant

        spixel_invcov = spixel_invcov_.reshape(1, -1, 9).permute(0,2,1).contiguous()
        spixel_mu = spixel_mu.permute(-1,-2).unsqueeze(0)
        cmix = cmix.T.unsqueeze(0)

        if if_timing:
            time_inv += time.time() - st_inv

        if spixel_invcov.isnan().sum()>0 or spixel_mu.isnan().sum()>0:
            print('ssn_3d_iter(): nan happens in spixel_invcov or spixel_mu !')
            import ipdb; ipdb.set_trace()

    if if_timing:
        # print(f'ssn_3d iteration took : {(time.time()-st)*1000:03f} ms for {n_iter:d} iters') 
        print(f'ssn_3d module: time_E={time_E*1000:03f}, time_M={time_M*1000:03f}, time_regu={time_regu*1000:03f}, time_rel2abs={time_rel2abs*1000:03f}, time_det={time_det*1000:03f}, time_inv={time_inv*1000:03f}')

    #hard_labels: 9xHxW, the idx for adjacent superpixels. indx 4 is the center pixel
    # 3x3 index for local neighborhood indexing:
    # 0 1 2 
    # 3 4 5
    # 6 7 8
    # An intuitive way to think about hard_labels is: for each pixel [ir, ic],
    # it is 'explained' by 9 GMMs close to it. hard_labels[:, ir, ic] saves the
    # index of these GMMs
    #TODO: !! hard_labels-related seems to be buggy!! --> we need to get abs_affinity using the hard_labels!

    if use_indx_init:
        hard_labels = abs_indices[1, :].reshape(-1, height, width)
    else:
        hard_labels = get_hard_abs_labels(affinity_matrix,
                                          abs_indices[1, :].reshape(-1, height, width),
                                          num_spixels_width)

    #bound hard labels
    if bound_hard_labels:
        out_bound_mask_ =torch.logical_or(hard_labels>= num_spixels, hard_labels<0,)
        hard_labels[out_bound_mask_] =\
            hard_labels[4].unsqueeze(0).expand(9, hard_labels.shape[1], hard_labels.shape[2])[out_bound_mask_]
    
    #update abs_affinity using hard_labels (9 H W)#
    # abs_affinity = affinity_rel2abs(affinity_matrix, None, num_spixels, hard_labels =None) #B J N_pix

    if return_affinity_mat:
        return abs_affinity, affinity_matrix, mix, spixel_mu, spixel_cov.reshape(-1,3,3), hard_labels # we do not need dist_matrix
    else:
        if return_abs_ind:
            return abs_affinity, dist_matrix, mix, spixel_mu, spixel_cov.reshape(-1,3,3), hard_labels
        else:
            return abs_affinity, dist_matrix, mix, spixel_mu, spixel_cov.reshape(-1,3,3), None

@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_maps, num_spixels_width):
    '''
    INPUT:
    affinity_matrix: B 9 (HW)
    init_label_maps: 9 H W, initial spixel labels for all pixels
    num_spixels_width: scalar - number of pixel in the width direction
    '''

    #relative_label: Bx(HW) size. This may change the spixel index for one
    #pixel, based on the gamma value. As a result, the relative label will be
    #different from the initial spixel assignment for the central pixel 
    relative_label = affinity_matrix.max(1)[1] 

    H, W = init_label_maps.shape[-2], init_label_maps.shape[-1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_maps + relative_spix_indices[relative_label].reshape(-1, H, W)
    return label.long()
