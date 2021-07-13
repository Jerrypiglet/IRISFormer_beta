'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''

import torch
from torch.functional import Tensor
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import models_def.model_nvidia.utils_nvidia.models as model_utils
import models_def.model_nvidia.utils_nvidia.misc as m_misc
import models_def.model_nvidia.align as align
import models_def.model_nvidia.raycaster as raycaster
from models_def.model_nvidia.raft.raft import RAFT

from matplotlib.pyplot import *

class AppGMM(torch.nn.Module):
    '''
    fusion net and pose transformer
    '''

    def __init__(
            self, args, ):
        '''
        '''
        super().__init__()
        self.learning_rate = args.cfg.MODEL_GMM.learning_rate
        self.model_optical_flow = None
        self.cam_intrinsic = None
        self.ssn_grid_spixel= args.cfg.MODEL_GMM.ssn_grid_spixel
        self.src_idx= args.cfg.MODEL_GMM.src_idx

        # self.spixel_nums =  (21, 15)  #w, h
        self.spixel_nums =  (32, 24)  #w, h
        self.J = self.spixel_nums[0]*self.spixel_nums[1] 

        self.total_step=0
        self.grad_clip = args.cfg.MODEL_GMM.grad_clip
        self.status_pred_in = None
        self.init_inbound_mask = None

        self.set_optical_flow_model(args)

        self.opt = args

    def configure_optimizers(self):
        '''
		configure the optimizers here
		Example:

        optimizer = torch.optim.Adam(
                list(self.model1.parameters()) + \
                        list(self.model2.parameters())  + \
                        list(self.model3.parameters()), 
                lr=self.learning_rate) 

        return optimizer
        '''
        pass

    def forward(self, batch, batch_idx, return_dict=False, res_dir=None):
        '''
        INPUTS:
        batch: dict of 
        imgs_ref -      nbatch x 3 x H x W
        imgs_src -      nbatch x N_src*3 x H x W
        dmaps_ref-      nbatch x 1 x H x W
        dmaps_src -     nbatch x N_src x H x W
        src_cam_poses - nbatch x N_src x 4 x 4
        '''

        # If ture, the spixel index assignment for all pixels would be the inital grid initialization
        ssn_grid_spixel = self.ssn_grid_spixel

        imgs_ref = batch['imgs_ref'].cuda()
        # imgs_src = batch['imgs_src'][:, self.src_idx * 3:(self.src_idx+1)*3, ...].cuda()
        # imgs_src = batch['imgs_src'].cuda()
        dmaps_ref_gt = batch['dmaps_ref'].cuda()
        # last_frame_traj = batch['last_frame_traj']

        J = self.spixel_nums[0] * self.spixel_nums[1]
        H, W = imgs_ref.shape[-2], imgs_ref.shape[-1]
        # T_pred_gt = batch['src_cam_poses'][0, 1].cuda().unsqueeze(0)  # assuming t_win=2
        # T_pred_gt_inv = T_pred_gt.inverse()
        # flow_mask, flow_prev2ref, flow_ref2prev=None, None, None
        # dmap_resample = None

        # invalid frame in ScanNet (all zero depth map or invalid pose)
        # if self.training and ((dmaps_ref_gt.max() == 0 and dmaps_ref_gt.min() == 0) or T_pred_gt.isnan().sum() > 0):
        #     self.status_pred_in = None
        #     return None

        #inpainting dmap_gt
        # dmaps_ref_gt = m_misc.inpaint_depth(dmaps_ref_gt)

        #for demo, use gt depth, and gt poses
        dmap_ref_meas = dmaps_ref_gt
        # T_pred_gt_rescale = T_pred_gt
        # T_pred_gt_rescale_inv = T_pred_gt_rescale.inverse()

        if self.init_inbound_mask is None:
            self.init_inbound_mask = m_misc.get_init_inbound_masks(
                H, W, self.spixel_nums)  # 9xHxW
            self.init_inbound_mask.requires_grad = False

        # for demo purpose, we will do the fusion of resampled value and noisy measured values
        dmap_update = dmaps_ref_gt

        #dmap_update to 3DGMM
        reg1 = 1e-3  
        weights = (dmaps_ref_gt > 0).to(dmap_ref_meas).reshape( 1, -1) 

        # print(dmap_update.shape)
        batch_size = dmap_update.shape[0]
        gamma_update_list = []
        for sample_idx in range(batch_size):
            mix_update, mu_update, cov_update, gamma_update, gamma_rel_update, abs_spixel_ind_update = \
                align.depth2gmm(
                    dmap_update[sample_idx:sample_idx+1],
                    [self.cam_intrinsic],
                    ssn_iter=10,
                    spixel_dim=self.spixel_nums,
                    reg1=reg1,
                    reg2=0,
                    normalize=False,
                    Weights=weights,
                    return_pcd=False,
                    return_rel_affinity=True,
                    bound_abs_spixel_ind=True,
                    use_indx_init=ssn_grid_spixel)

            gamma_update_list.append(gamma_update)

            # print(gamma_update.shape, gamma_rel_update.shape, abs_spixel_ind_update.shape) # [1, 315, 240, 320], [1, 9, 76800], [9, 240, 320]
            # print(abs_spixel_ind_update[:, 0, 0])
            # print(gamma_update[0, :, 0, 0])

            cluster_t_valid = torch.zeros(self.J).bool().cuda()
            in_bound_mask_warp = torch.zeros([9, H, W]).bool().cuda()
            cluster_t_valid[:] = True
            in_bound_mask_warp[:] = True

            # dmap_update_resample, sigma_update_resample = \
            #     self.gmm2depth(
            #         mix_update.squeeze(),
            #         mu_update.squeeze(),
            #         cov_update.squeeze(),
            #         cluster_t_valid,
            #         abs_spixel_ind_update,
            #         torch.ones_like(in_bound_mask_warp),
            #         gamma_update.reshape(-1, H*W).T, )

            # save the depth maps here for demonstration purpose #

            # diff_depth = dmaps_ref_gt.squeeze() - dmap_update_resample.squeeze()
            # fldr = self.opt.summary_vis_path_task
            # m_misc.msavefig(dmap_update_resample.squeeze(), f'{fldr}/depth_resampled_{batch_idx:04d}.png', vmin=0, vmax=5)
            # m_misc.msavefig(dmaps_ref_gt.squeeze(), f'{fldr}/depth_gt_{batch_idx:04d}.png', vmin=0, vmax=5)
            # m_misc.msavefig(diff_depth.squeeze().abs(), f'{fldr}/depth_diff_{batch_idx:04d}.png', vmin=0, vmax=1)
            # print(f'save res to {fldr}/depth_diff_{batch_idx:04d}.png..')



        if not return_dict:
            return dmap_update
        else: # return a dict
            # if flow_mask is not None:
            #     flow_mask = flow_mask.unsqueeze(0).unsqueeze(0)
            res = dict({
                # 'gmm_params_update': [mix_update, mu_update, cov_update],
                'gamma_update': torch.cat(gamma_update_list, 0), 
                # 'flow_mask': flow_mask,
                # 'flow_ref2prev': flow_ref2prev,
                # 'flow_prev2ref': flow_prev2ref,
            })

            #'diff_flow_norm_loss': diff_flow_norm_loss,
            return  res

    def training_step(self, batch, batch_idx):
        '''
        forward pass and calculate loss
        '''

        res = self.forward(batch, batch_idx, return_dict=True)

        if res is None:
            return None  # invalid iteration

        imgs_ref = batch['imgs_ref'].cuda()
        # imgs_src = batch['imgs_src'][:, self.src_idx *
        #                              3:(self.src_idx+1)*3, ...].cuda()
        dmaps_ref = batch['dmaps_ref'].cuda()

        # dmaps_src = batch['dmaps_src'][:, self.src_idx, ...].unsqueeze(1).cuda()
        # flow_mask = res['flow_mask']
        # flow_ref2prev = res['flow_ref2prev']

        ##Get the loss here..
        loss=0.
        ##

        output_dict = {'output_GMM': res}

        return loss, output_dict

    def set_to_test(self):
        self.eval()
        self.model_optical_flow.eval()

    def set_to_train(self):
        self.train()
        self.model_optical_flow.eval()

    def test_step(self, res_dir, batch, batch_idx):
        '''
        similar to train_step()
        '''

        # fill in your test code here...
        #
        return None

    def set_camintrinsic(self, cam_intrinsic):
        self.cam_intrinsic = cam_intrinsic

    def set_optical_flow_model(self, args,):
        # load the optical flow net

        self.model_optical_flow = torch.nn.DataParallel(RAFT(args).cuda())
        self.model_optical_flow.load_state_dict(torch.load(args.cfg.MODEL_GMM.RAFT.OF_model_path))
        self.model_optical_flow.eval()

    def set_logger(self, args,log_dir):
        '''set logger'''
        ret = m_misc.m_makedir(log_dir)
        if not ret:  # log_dir path already there, we will created new copy
           from datetime import datetime
           today = datetime.now()
           log_dir = log_dir + today.strftime('%Y%m%d-%H%M%S')
           ret = m_misc.m_makedir(log_dir)

        args_str = model_utils.save_args(
            args, f'{log_dir}/tr_paras.txt')
        self.log_dir = log_dir
        self.logger = SummaryWriter(log_dir=log_dir)

        tr_paras_str = ''
        for arg_str in args_str:
            tr_paras_str += '%s     \n' % (arg_str)
        self.logger.add_text('tr paras', tr_paras_str,)

    def mlog_img(self, img, title, vmax=None, vmin=None, ):
        #log image
        if vmax is not None:
            img = torch.clamp(img, max=vmax, min=vmin)
        img = (img - img.min()) / (img.max() - img.min())
        self.logger.add_image(title, img, self.total_step)

    def mlog_scalar(self, name, scalar, ):
        #log scalar
        self.logger.add_scalar(name, scalar, self.total_step)

    def appearance_recon(self, gamma, feat_map, scale_feat_map=1):
        '''
        gamma: Q, BxJxHxW
        feat_map: BxDxHxW, where N*scale_feat_map=HW
        scale_feat_map
        '''
        if scale_feat_map != 1:
            # print(gamma.shape, feat_map.shape, scale_feat_map)
            gamma_resized = F.interpolate(gamma, scale_factor=1./float(scale_feat_map))
            gamma = gamma_resized
            # print(gamma_resized.shape, feat_map.shape)
            
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


    def gmm2depth(self, mix_resample,
                  mu_resample, cov_resample,
                  cluster_t_valid,
                  abs_spixel_ind_cur_warp,
                  in_bound_mask_warp,
                  gamma_cur,):
        '''
        resampling

        gmm to depth 

        INPUTS:
        cluster_t_valid - J ele bool array, only J_valid<J are set to True
        gamma_cur - N x J 

        optional 
        gmm_cur_rel - Nx 9

        OUTPUTS:
        dmap_resample, var_resample
        '''


        # Some GMMs are not valid due to (1) warped outside of view - detected in cluster_t_valid, (2) have 0 det (too flat in 3D)
        # We will not consider the invalid gmms
        idx_valid = cov_resample.det() > 0.

        cluster_t_valid_0 = cluster_t_valid.clone()
        cluster_t_valid_0[torch.nonzero(cluster_t_valid_0, as_tuple=False).squeeze()[
            torch.logical_not(idx_valid)]] = False
        #bound abs_spixel_ind
        out_bound_mask = torch.logical_or(torch.logical_or(
            abs_spixel_ind_cur_warp >= self.J, abs_spixel_ind_cur_warp < 0,), in_bound_mask_warp < .9)
        abs_spixel_ind_cur_warp[out_bound_mask] = abs_spixel_ind_cur_warp[4].unsqueeze(0).expand(
            9, abs_spixel_ind_cur_warp.shape[1], abs_spixel_ind_cur_warp.shape[2])[out_bound_mask]

        hash_indx_2_valid_indx = torch.cumsum(cluster_t_valid_0, dim=0) - 1
        indx_maps_hash = hash_indx_2_valid_indx[abs_spixel_ind_cur_warp]
        indx_maps_hash[indx_maps_hash < 0] = 0

        #--mu and sigma along the rays--#
        try:
            _, _, _, dmap_resample, sigma_resample = raycaster.ray_from_gmm(
                self.cam_intrinsic['unit_ray_array_2D'].T,
                mix_resample[idx_valid],
                mu_resample[idx_valid, :],
                cov_resample[idx_valid, :, :],
                imsize=in_bound_mask_warp.shape[-2:],
                nspixel=self.J,
                abs_spixel_ind=indx_maps_hash,
                gamma=gamma_cur[:, cluster_t_valid][:, idx_valid],
                reduce_method='direct_weight_sum',
                correction_factor=1e8,  # this parameter will not be used for 'direct_weight_sum'
                return_inv_sigma=False)
            return dmap_resample, sigma_resample
        except:
            print('ERR DURING RESAMPLING DEPTH')
            return None, None

