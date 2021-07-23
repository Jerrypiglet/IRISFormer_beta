'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''

from numpy.lib import utils
import torch
from torch.functional import Tensor
import torch.nn.functional as funct
from tensorboardX import SummaryWriter

import utils.models as model_utils
import utils.misc as m_misc
import model.align as align, model.raycaster as raycaster

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
        self.learning_rate = args.learning_rate
        self.model_optical_flow = None
        self.cam_intrinsic = None
        self.ssn_grid_spixel= args.ssn_grid_spixel
        self.src_idx= args.src_idx

        self.spixel_nums =  (21, 15)  #w, h
        self.J = self.spixel_nums[0]*self.spixel_nums[1] 

        self.total_step=0
        self.grad_clip = args.grad_clip
        self.status_pred_in = None
        self.init_inbound_mask = None

        self.set_optical_flow_model(args)

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
        imgs_src = batch['imgs_src'][:, self.src_idx * 3:(self.src_idx+1)*3, ...].cuda()

        imgs_src_all = [batch['imgs_src'][:, src_idx * 3:(src_idx+1)*3, ...].cuda() for src_idx in range(4)]

        dmaps_ref_gt = batch['dmaps_ref'].cuda()
        last_frame_traj = batch['last_frame_traj']

        J = self.spixel_nums[0] * self.spixel_nums[1]
        H, W = imgs_ref.shape[-2], imgs_ref.shape[-1]
        T_pred_gt = batch['src_cam_poses'][0, 1].cuda().unsqueeze(0)  # assuming t_win=2
        T_pred_gt_inv = T_pred_gt.inverse()
        flow_mask, flow_prev2ref, flow_ref2prev=None, None, None
        dmap_resample = None
        dmaps_src_gt_raw = batch['dmaps_src'].cuda()

        #inpainting dmap_gt
        dmaps_ref_gt = m_misc.inpaint_depth(dmaps_ref_gt)
        dmaps_src_gt = []
        for i in range(dmaps_src_gt_raw.shape[1]):
            dmaps_src_gt.append(m_misc.inpaint_depth(dmaps_src_gt_raw[0, i]).unsqueeze(0).unsqueeze(0))
        dmaps_src_gt= torch.cat(dmaps_src_gt, dim=0)
        

        #--- SSN 2D DEMO ---#
        #-----------#
        import ssn.ssn as ssn2d

        n_iter_ssn_2d = 10
        import time
        st=time.time()
        abs_affinity, dist_matrix, spixel_features = ssn2d.ssn_iter(
            dmaps_src_gt.repeat(8, 1,1,1), #32 channels raw img size depth map #imgs_ref , #imgs_ref + imgs_ref_add
            n_iter=n_iter_ssn_2d, 
            num_spixels_width=self.spixel_nums[0], 
            num_spixels_height=self.spixel_nums[1],
            index_add=False,
        )
        ed=time.time()
        print(f'ssn_2d batched took {(ed-st)*1000:.4f} ms')

        st=time.time()
        abs_affinity_cmp, dist_matrix_cmp, spixel_features_cmp = ssn2d.ssn_iter(
            dmaps_src_gt.repeat(8, 1,1,1), #imgs_ref , #imgs_ref + imgs_ref_add
            n_iter=n_iter_ssn_2d, 
            num_spixels_width=self.spixel_nums[0], 
            num_spixels_height=self.spixel_nums[1],
            index_add=True,
        )
        ed=time.time()
        print(f'ssn_2d batched took {(ed-st)*1000:.4f} ms')

        gamma_map = abs_affinity.reshape(abs_affinity.shape[0], abs_affinity.shape[1], H, W) # B J H W
        gamma_map_cmp = abs_affinity_cmp.reshape(abs_affinity_cmp.shape[0], abs_affinity_cmp.shape[1], H, W) # B J H W

        # boundary image from the previous version using sparse_coo_tensor
        img_boundary0 = m_misc.mark_gamma_boundary(
            img_rgb= imgs_src_all[0].squeeze().permute(1,2,0), 
            gamma = gamma_map[0].permute(1,2,0)) 

        # boundary image from the current version without using sparse_coo_tensor
        img_boundary1 = m_misc.mark_gamma_boundary(
            img_rgb= imgs_src_all[0].squeeze().permute(1,2,0), 
            gamma = gamma_map_cmp[0].permute(1,2,0))

        # import ipdb; ipdb.set_trace()
        #-----------#
        #-----------#





        #--- SSN 3D DEMO ---#
        #-----------#

        # invalid frame in ScanNet (all zero depth map or invalid pose)
        if self.training and ((dmaps_ref_gt.max() == 0 and dmaps_ref_gt.min() == 0) or T_pred_gt.isnan().sum() > 0):
            self.status_pred_in = None
            return None


        #for demo, use gt depth, and gt poses
        dmap_ref_meas = dmaps_ref_gt
        T_pred_gt_rescale = T_pred_gt
        T_pred_gt_rescale_inv = T_pred_gt_rescale.inverse()

        if self.init_inbound_mask is None:
            self.init_inbound_mask = m_misc.get_init_inbound_masks(
                H, W, self.spixel_nums)  # 9xHxW
            self.init_inbound_mask.requires_grad = False

        if self.status_pred_in is not None:
            #== Get relative camera pose from prev to ref ==#
            imgs_prev = self.status_pred_in['imgs_ref']
            dmap_prev = self.status_pred_in['dmap_update']
            mix_prev, mu_prev, cov_prev, gamma_previous_log, abs_spixel_ind_prev = \
                self.status_pred_in['mix_update'], self.status_pred_in['mu_update'], \
                self.status_pred_in['cov_update'], self.status_pred_in['gamma_update'], \
                self.status_pred_in['abs_spixel_ind']

            OF_iter = 10 #optical flow maximal iteration for RAFT

            gmm_param_cur, gamma_cur, abs_spixel_ind_cur_warp, in_bound_mask_warp, flow_ref2prev = \
                align.get_src_gmm(
                    img_ref=imgs_ref,
                    abs_affinity_src=gamma_previous_log,
                    img_src=imgs_src,
                    dmap_ref=dmap_ref_meas,
                    model_optical_flow=self.model_optical_flow,
                    Cam_Intrinscis_list=[self.cam_intrinsic],
                    abs_spixel_ind_src=abs_spixel_ind_prev.unsqueeze(0),
                    in_bound_map=self.init_inbound_mask.unsqueeze(0),
                    num_spixels_width=self.spixel_nums[0],
                    num_spixels_height=self.spixel_nums[1],
                    OF_iter=OF_iter,
                )

            in_bound_mask_warp = in_bound_mask_warp.T.contiguous().reshape(9, H, W)  # 9xHxW
            _, flow_prev2ref, _, _, _ = self.model_optical_flow(
                2 * imgs_src - 1,
                2 * imgs_ref - 1,
                iters=OF_iter,
                test_mode=True)
            flow_prev2ref = flow_prev2ref.detach()

            #get the flow mask by forard-backward warping of optical flow check#
            # useful in dealing with occlusions etc.#
            diff_norm_ref = align.check_optical_flows(
                flow_ref2prev, flow_prev2ref).squeeze()
            diff_norm_prev = align.check_optical_flows(
                flow_prev2ref, flow_ref2prev).squeeze()
            flow_mask = diff_norm_ref < 5.  
            flow_mask_src2prev = diff_norm_prev < 5. 

            # do the warping for sanity check#
            img_prev_warp= align.warp_img_from_flow_v1(imgs_ref, flow_prev2ref)
            img_ref_warp= align.warp_img_from_flow_v1(imgs_prev, flow_ref2prev)

            '''get global pose'''
            global_pose = T_pred_gt_inv.unsqueeze(0)  # 1x1x4x4
            global_pose_inv = T_pred_gt.unsqueeze(0)

            # 1xJx4x4, gmm T_j from prev to ref
            Ts_align_flow = global_pose.repeat([1, J, 1, 1])
            Ts_align_flow_inv = global_pose_inv.repeat(
                [1, J, 1, 1])  # gmm T_j from ref to prev

            '''Resample'''
            # Resample depth given transformed GMMs#
            abs_spixel_ind_cur_warp = abs_spixel_ind_cur_warp.T.contiguous().reshape(9, H, W)  # 9xHxW
            cluster_t_valid = gamma_cur.sum(0).squeeze() > 10

            mix_pred_Ts, mu_pred_Ts, cov_pred_Ts = align.transformGMM_Rt(
                mix_prev.squeeze()[cluster_t_valid],
                mu_prev.squeeze()[cluster_t_valid],
                cov_prev.squeeze()[cluster_t_valid],
                R=Ts_align_flow[0, cluster_t_valid, :3, :3],
                t=Ts_align_flow[0, cluster_t_valid, :3, 3])
            mix_resample, mu_resample, cov_resample = mix_pred_Ts, mu_pred_Ts, cov_pred_Ts

            dmap_resample, sigma_resample = \
                self.gmm2depth(
                    mix_resample,
                    mu_resample,
                    cov_resample,
                    cluster_t_valid,
                    abs_spixel_ind_cur_warp,
                    in_bound_mask_warp,
                    gamma_cur, # NXJ
                    )

            if dmap_resample is None: # error happes during resampling..
                self.status_pred_in = None

                # NOTE: set the break point here when resampling encounters an error, useful for debugging training, since 
                # during training the depth map and pose are not accurate, that will lead strange values for the depth sampler
                # uncomment this line if you want to have robust training where we will ignore the loss and gradient when resampling is not working
                # for some frames
                import ipdb; ipdb.set_trace() 

                return None

            #threshold the resampled dmap to be larger than zero#
            dmap_resample[dmap_resample < 0.] = 0.


        # for demo purpose, we will do the fusion of resampled value and noisy measured values
        dmap_update = dmaps_ref_gt

        #for demo purpose: we will compare the resapmled depth with the GT depth
        if dmap_resample is not None:
            diff_depth = dmaps_ref_gt.squeeze() - dmap_resample.squeeze()
            # save the depth maps here for demonstration purpose #
            import os
            fldr='../res/ssn-resample-demo'
            os.makedirs(fldr, exist_ok=True)
            m_misc.msavefig(dmap_resample.squeeze(), f'{fldr}/depth_resampled_{batch_idx:04d}.png', vmin=0, vmax=5)
            m_misc.msavefig(dmaps_ref_gt.squeeze(), f'{fldr}/depth_gt_{batch_idx:04d}.png', vmin=0, vmax=5)
            m_misc.msavefig(diff_depth.squeeze().abs(), f'{fldr}/depth_diff_{batch_idx:04d}.png', vmin=0, vmax=1)
            print(f'save res to {fldr}/depth_diff_{batch_idx:04d}.png..')

        #dmap_update to 3DGMM
        reg1 = 1e-3  
        weights = (dmaps_ref_gt > 0).to(dmap_ref_meas).reshape( 1, -1) 
        mix_update, mu_update, cov_update, gamma_update, gamma_rel_update, abs_spixel_ind_update = \
            align.depth2gmm(
                dmap_update,
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

        #demo: gmm2depth
        if batch_idx>=1: # not the first frame
            cluster_t_valid[:]=True
            dmap_update_resample, sigma_update_resample = \
                self.gmm2depth(
                    mix_update.squeeze(),
                    mu_update.squeeze(),
                    cov_update.squeeze(),
                    cluster_t_valid,
                    abs_spixel_ind_update,
                    torch.ones_like(in_bound_mask_warp),
                    gamma_update.reshape(-1, H*W).T, )


        if last_frame_traj:
            self.status_pred_in = None  # prepare for the next trajectory
            self.depth_max_threshold = None
        else:
            self.status_pred_in = {
                'imgs_ref':       imgs_ref.detach(),
                'mix_update':     mix_update.detach(),
                'mu_update':      mu_update.detach(),
                'cov_update':     cov_update.detach(),
                'gamma_update':   gamma_update.detach(),
                'dmap_update':    dmap_update.detach(),
                'abs_spixel_ind': abs_spixel_ind_update.detach(),
            }

        if not return_dict:
            return dmap_update
        else: # return a dict
            if flow_mask is not None:
                flow_mask = flow_mask.unsqueeze(0).unsqueeze(0)
            res = dict({
                'gmm_params_update': [mix_update, mu_update, cov_update],
                'flow_mask': flow_mask,
                'flow_ref2prev': flow_ref2prev,
                'flow_prev2ref': flow_prev2ref,
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
        imgs_src = batch['imgs_src'][:, self.src_idx *
                                     3:(self.src_idx+1)*3, ...].cuda()
        dmaps_ref = batch['dmaps_ref'].cuda()

        dmaps_src = batch['dmaps_src'][:, self.src_idx, ...].unsqueeze(1).cuda()
        flow_mask = res['flow_mask']
        flow_ref2prev = res['flow_ref2prev']

        ##Get the loss here..
        loss=0.
        ##

        return loss

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
        from raft.raft import RAFT

        self.model_optical_flow = torch.nn.DataParallel(RAFT(args))
        self.model_optical_flow.load_state_dict(torch.load(args.OF_model_path))
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
        mix_resample - J elements vector
        mu_resample - Jx3, each row is the mean of one cluster
        cov_resample - Jx3x3, the covariance matrix of one cluster
        cluster_t_valid - J ele bool array, only J_valid<J are set to True
        abs_spixel_ind_update -9xHxW, the local hashing map got from depth2gmm
        gamma_cur - N x J  - the gamma matrix from depth2gmm, N=H*W
        in_bound_mask_warp - 9xHxW, for single-frame, set to all one (ones(9,H,W))

        optional 
        gmm_cur_rel - Nx 9

        OUTPUTS:
        dmap_resample, var_resample
        '''


        # Some GMMs are not valid due to (1) warped outside of view - detected in cluster_t_valid, (2) have 0 det (too flat in 3D)
        # We will not consider the invalid gmms
        idx_valid = cov_resample.det() > 0.

        cluster_t_valid_0 = cluster_t_valid.clone()
        cluster_t_valid_0[torch.nonzero(cluster_t_valid_0, as_tuple=False).squeeze()[ torch.logical_not(idx_valid)]] = False
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

