'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''
import torch 
import torch.nn.functional as F
import models_def.model_nvidia.ssn_3d.ssn_3d as ssn_3d 
import numpy as np

def uv_coord_delta_to_image(uv_coord_delta, img, uv_center, focal_length=None, mode='bilinear', padding_mode='zeros', debug = False):
    '''
    Sample from src image (img), given the optical flow from the ref image

    INPUT

    uv_coord_delta - NCHW optical flow 
    uv_coord_delta = uv_coord_ref - uv_coord_src from back_warp_th_Rt(), 

    img - NCHW, src image  

    uv_center - [ucenter, vcenter]

    The unit for the delta coords is pixel. So uv_center is needed for image domain coordinate normalization

    OUTPUT

    img_interp - NCHW, the interpolated image same size as input img;

    uv_coords  - NCHW, C=2, the normalized uv coordinates for interpolation
    
    '''
    H, W = img.shape[-2], img.shape[-1] 
    H_v = torch.linspace(1, H, H, device=img.device) - .5
    W_v = torch.linspace(1, W, W, device=img.device) - .5 
    v_coord_ref, u_coord_ref = torch.meshgrid( H_v,  W_v ) #TODO: input this value so that we don't need to build it every time
    uv_coord_ref = torch.stack(( u_coord_ref, v_coord_ref )).unsqueeze(0)
    uv_coord_src = uv_coord_ref - uv_coord_delta # in pixels
    u_coords, v_coords = uv_coord_src[:, 0, :, :], uv_coord_src[:, 1, :, :]
    u_center, v_center = uv_center[0], uv_center[1]
    u_coords_ = (u_coords - u_center) / torch.clamp(u_center, min=1e-6) 
    v_coords_ = (v_coords - v_center) / torch.clamp(v_center, min=1e-6)
    uv_coords_ = torch.cat( (u_coords_.unsqueeze_(3), v_coords_.unsqueeze_(3)) , dim=3) 
    img_interp = F.grid_sample( img, uv_coords_, mode=mode, align_corners= True, padding_mode=padding_mode)

    if focal_length is not None:
        u_coords, v_coords = (u_coords - u_center) / torch.clamp(focal_length[0], min=1e-6), \
                             (v_coords - v_center) / torch.clamp(focal_length[1], min=1e-6)
        uv_coords = torch.cat( (u_coords.unsqueeze_(3), v_coords.unsqueeze_(3)) , dim=3)
        return img_interp, uv_coords.permute(0, 3, 1, 2).contiguous()
    else:
        return img_interp, None

def dmaps_to_pcd(dmaps, Cam_intrin, inverse_depth =False, return_nonzero = False):
    '''dmaps to pcd
    dmaps - NCHW , C=1
    Cam_intrin - Cam_intrin obj (dict) or a torch array
    return_nonzero - only return nonzero values
    '''
    pcds, npts, indx_nonzero = [], [], []

    if isinstance(Cam_intrin, dict):
        unitray_array = Cam_intrin['unit_ray_array_2D']
    else:
        unitray_array = Cam_intrin

    for i in range(dmaps.shape[0]):
        dmap = dmaps[i].squeeze()
        if inverse_depth:
            pcd = 1./torch.clamp(dmap.view(1, dmap.numel()), min=1e-6) * unitray_array.to(dmaps)
        else:
            pcd = dmap.view(1, dmap.numel()) *unitray_array.to(dmaps)

        if return_nonzero:
            indx = torch.nonzero( pcd[2,:], as_tuple=True )
            pcd = pcd[:, indx[0] ]
            indx_nonzero.append( indx[0] )
            pcds.append(pcd.transpose(0,1))
            npts.append( len(indx[0]) )
        else:
            pcds.append(pcd.transpose(0,1))
            npts.append(pcd.shape[1]) 

    pcds = torch.cat(pcds, dim=0)

    if not return_nonzero:
        return pcds, npts
    else:
        return pcds, npts, indx_nonzero

def _Mstep(gamma, pts, weights=None, reg=0):
    outer_pts = ssn_3d.outer_prod(pts, pts).reshape(-1,9)
    mu, sigma, pi = \
            ssn_3d.M_step(
                    gamma.permute(1,0), 
                    pts, pts_outer=outer_pts, 
                    weights = weights.permute(1,0) if weights is not None else None, 
                    robust_divide=True, epsilon=1e-5)
    # import ipdb; ipdb.set_trace()
    pi = pi.T
    sigma = sigma.reshape(-1,3,3)
    if reg>0:
        sigma = ssn_3d.regularzie_cov(sigma, regularizer=reg).unsqueeze(0)
    else:
        sigma = sigma.unsqueeze(0)

    mu = mu.unsqueeze(0)

    return pi, mu, sigma

def get_src_gmm(
    img_ref, abs_affinity_src,
    img_src, dmap_ref,
    model_optical_flow, 
    Cam_Intrinscis_list, 
    abs_spixel_ind_src =None, 
    in_bound_map = None,
    flow_ref2src_in = None, 
    num_spixels_height=16, num_spixels_width=16, 
    OF_iter=20, reg1=1e-2, reg2=0, ):

    #TODO: combine this function with depth2gmm() in this file, there are some shared code..

    num_spixels = num_spixels_height * num_spixels_width
    feat_pcd_ref, _,  = dmaps_to_pcd(dmap_ref, Cam_Intrinscis_list[0], inverse_depth=False, return_nonzero=False) 

    # import time
    # st = time.time()
    if flow_ref2src_in is None:
        # optical flow ref->src
        #flow_ref2src : optical flow FROM ref to src. The mesh grid is on ref view. output is viewed from the ref view
        _, flow_ref2src, _, _, _ = model_optical_flow( 2* img_ref - 1, 2* img_src - 1, iters= OF_iter, test_mode=True) 
        flow_ref2src = flow_ref2src.detach()
    else:
        flow_ref2src = flow_ref2src_in
    # print(f'---flow module : {(time.time()-st)* 1000.:.3f} ms')

    # get gamma for the src view
    uv_center = torch.FloatTensor([Cam_Intrinscis_list[0]['intrinsic_M'][0, 2], Cam_Intrinscis_list[0]['intrinsic_M'][1, 2] ])
    focal_len = torch.FloatTensor([Cam_Intrinscis_list[0]['intrinsic_M'][0, 0], Cam_Intrinscis_list[0]['intrinsic_M'][1, 1] ])

    imgs_gamma_ref_corresp, _ = \
            uv_coord_delta_to_image(
            -flow_ref2src.detach(), 
            abs_affinity_src, 
            uv_center=uv_center, 
            focal_length=focal_len, 
            padding_mode='reflection', # 'reflection', 'zeros'
            debug=False)

    with torch.no_grad():
        if abs_spixel_ind_src is not None:
            assert in_bound_map is not None

            imgs_abs_spix_ref_corresp, _ = \
                    uv_coord_delta_to_image(
                    -flow_ref2src.detach(), 
                    abs_spixel_ind_src.float(),
                    uv_center = uv_center, 
                    focal_length = focal_len, 
                    mode ='nearest', #we choose nearest since we are interpolating index values (abs_spixel_ind_src) and booleans (in_bound_map)
                    padding_mode = 'reflection',
                    debug =False)

            imgs_in_bound, _ = \
                    uv_coord_delta_to_image(
                    -flow_ref2src.detach(), #NCHW, C=2
                    in_bound_map,#N9HW
                    uv_center = uv_center,#2-ele 
                    focal_length = focal_len,#2-ele 
                    mode ='nearest', # we choose nearest since we are interpolating index values (abs_spixel_ind_src) and booleans (in_bound_map)
                    padding_mode = 'zeros',
                    debug =False)


        else:
            imgs_abs_spix_ref_corresp = None 
            imgs_in_bound = None

    weights_ref = imgs_gamma_ref_corresp.sum(1).view(-1,1) 
    gamma_ref = imgs_gamma_ref_corresp.permute(0, 2, 3, 1).view(-1, num_spixels).contiguous()

    # get gmm parameters in the src view
    mix_ref, mu_ref, sigma_ref= _Mstep(gamma_ref, feat_pcd_ref, weights=weights_ref, reg= reg1) 
    gmm_params_ref = [mix_ref, mu_ref, sigma_ref]


    if abs_spixel_ind_src is not None:
        return  gmm_params_ref, \
                gamma_ref, \
                imgs_abs_spix_ref_corresp.permute(0, 2, 3, 1).view(-1, 9).contiguous().long(),\
                imgs_in_bound.permute(0, 2, 3, 1).view(-1, 9).contiguous(), \
                flow_ref2src
                
    else:
        return  gmm_params_ref, \
                gamma_ref,\
                None, None, None


def depth2gmm(dmap_, Cam_Intrinscis_list,
              ssn_iter=5, spixel_dim=[27, 18], reg1=.01, reg2=0,
              normalize=False,  Weights=None,
              return_pcd=False,
              return_rel_affinity=False,
              bound_abs_spixel_ind=True,
              use_indx_init=False):
    '''from depth map to 3D GMMs
    inputs 
    dmap - NCHW, C=1
    Cam_Intrinscis_list - list of Cam_Intrinsics
    spixel_dim - [width, height], [18,27]
    reg1, reg2 - [.01] ,[0]
    normalize - if normalize dmap such that max depth value is 5.
    Weights - weight vector for pts, Nx(HxW)
    '''

    if not normalize:
        dmap = dmap_
    else: # normalize
        scale = 1/dmap_.max()*5.
        dmap = dmap_ * scale

    num_spixels_height,num_spixels_width = spixel_dim[1], spixel_dim[0] 
    num_spixels = spixel_dim[0]*spixel_dim[1] 

    feat_pcd_ref, npts_ref, = dmaps_to_pcd(dmap, Cam_Intrinscis_list[0], inverse_depth=False, return_nonzero=False)

    feat_pcd_ref_map = feat_pcd_ref.T.reshape(-1, *dmap.shape[-2:]).unsqueeze(0) # [b, 3, H, W]
    # print(feat_pcd_ref.shape, dmap.shape, feat_pcd_ref_map.shape)
    feat_ssn_in_ref  = feat_pcd_ref_map

    abs_affinity_ref, rel, _, _, _, abs_spixel_ind = \
            ssn_3d.ssn3d_iter(
                    feat_ssn_in_ref, n_iter= ssn_iter, 
                    num_spixels_width=num_spixels_width, 
                    num_spixels_height=num_spixels_height, 
                    reg = reg1, reg2=reg2,
                    return_abs_ind=True,
                    Weights = Weights,
                    return_affinity_mat= return_rel_affinity,
                    bound_hard_labels= bound_abs_spixel_ind,
                    use_indx_init= use_indx_init)

    abs_affinity_ref = abs_affinity_ref.view(feat_ssn_in_ref.shape[0], num_spixels, *feat_ssn_in_ref.shape[-2:]).contiguous()
    gamma_ref = abs_affinity_ref.permute(0, 2, 3, 1).view(-1, num_spixels).contiguous()

    mix, mu, sigma = _Mstep(gamma_ref, feat_pcd_ref, reg=reg1, weights = Weights.T if Weights is not None else None)

    #return
    if normalize:
        mu = mu / scale
        sigma = sigma / (scale**2)
        if return_pcd:
            return mix, mu, sigma, abs_affinity_ref, feat_pcd_ref, scale, abs_spixel_ind
        else:
            return mix, mu, sigma, abs_affinity_ref, scale, abs_spixel_ind
    else:
        if return_pcd:
            return mix, mu, sigma, abs_affinity_ref, feat_pcd_ref, abs_spixel_ind
        elif return_rel_affinity:
            return mix, mu, sigma, abs_affinity_ref, rel, abs_spixel_ind # [this one is being used]
        else:
            return mix, mu, sigma, abs_affinity_ref, abs_spixel_ind



def transformGMM_Rt(mix, mu, cov, R, t, unsqueeze=False):
    '''
    input:
    mix - not used
    mu - Jx3
    cov- Jx3x3

    R - 3x3
    t - 3

    or

    R - Jx3x3 
    t - Jx3

    In this case, each mu, cov will be transformed independently

    output:
    mix - same as input mix, not used inside 
    mu_out - Jx3
    cov - Jx3x3
    '''
    # mu_out = torch.matmul(R, mu.T).T + t 
    J = mu.shape[0]

    if R.ndim==2 and t.ndim==1:
        mu_out = torch.matmul(R.unsqueeze(0).expand(J, 3,3), mu.unsqueeze(-1)) + t.unsqueeze(-1)
        cov_out = torch.matmul(R.unsqueeze(0).expand(J,3,3), cov ).matmul(R.T.contiguous().unsqueeze(0).expand(J,3,3))
    elif R.ndim==3 and t.ndim==2:
        assert R.shape[0] == t.shape[0] == J
        mu_out = torch.matmul(R, mu.unsqueeze(-1)) + t.unsqueeze(-1) # Jx3x1
        cov_out = torch.matmul(R, cov).matmul(R.permute(0,2,1).contiguous()) # Jx3x3
    else:
        raise Exception('R and/or t is not of the correct dim!')

    if unsqueeze:
        return mix, mu_out.squeeze(-1).unsqueeze(0).contiguous(), cov_out.contiguous().unsqueeze(0)
    else:
        return mix, mu_out.squeeze(-1).contiguous(), cov_out.contiguous()
    
def get_rel_extrinsicM(ext_ref, ext_src):
    ''' Get the extrinisc matrix from ref_view to src_view 
    return:
    T_ref2src - transformation from ref view to src view
    P_src = T_ref2src @ P_ref
    where P_ref is the 3D point location wrt ref view; P_src is the same 3D point location wrt src view
    '''
    T_ref2src = ext_src.dot( np.linalg.inv( ext_ref))
    return T_ref2src

def check_optical_flows(flow_tar2src, flow_src2tar):
    '''
    flow_tar2src, flow_src2tar - N 2 H W
    output:
    diff_norm - NxHxW
    '''
    # def warp_img_from_flow( img_prev,  flow_cur2prev, ):  
    flow_src2tar_interp = warp_img_from_flow_v1(flow_src2tar,  flow_tar2src) #in the tar view
    diff_tarview = flow_tar2src + flow_src2tar_interp
    diff_norm= diff_tarview.norm(dim=1)
    return diff_norm

def warp_img_from_flow_v1( img_prev,  flow_cur2prev, ):
    '''
    warp image from prev view to ref view
    input:
    img_prev - NCHW

    output:
    img_cur_warp - NCHW, warped from prev view
    '''
    H, W = img_prev.shape[-2], img_prev.shape[-1]
    S = max(H/2., W/2.)
    uv_center = torch.FloatTensor([W/2., H/2.]) 
    img_cur_warp, _ = \
            uv_coord_delta_to_image(
            -flow_cur2prev.detach(), 
            img_prev,
            uv_center=uv_center, 
            debug=False)

    return img_cur_warp