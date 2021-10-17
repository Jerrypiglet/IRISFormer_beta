import torch
import torch.nn as nn
from torch.autograd import Variable
# import models
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import statistics
import torchvision.utils as vutils
from utils.utils_total3D.utils_OR_cam import get_rotation_matrix_gt, get_rotation_matix_result
from utils.utils_total3D.utils_OR_layout import get_layout_bdb_sunrgbd, reindex_layout
from utils.utils_total3D.utils_OR_visualize import Box

from models_def.models_brdf import LSregress

from models_def.losses_total3d import JointLoss, PoseLoss, DetLoss, ReconLoss, SVRLoss
from models_def.losses_total3d import emitter_cls_criterion_mean, emitter_cls_criterion_L2_mean, cls_criterion_mean, emitter_cls_criterion_L2_none

from utils.utils_total3D.net_utils_libs import get_bdb_evaluation, get_mask_status
from utils.utils_total3D.utils_OR_visualize import format_bboxes
from utils.utils_total3D.libs.tools import write_obj
from pathlib import Path
from icecream import ic


def get_labels_dict_layout_emitter(labels_dict_input, data_batch, opt):
    labels_dict = {'layout_labels': {}, 'object_labels': {}, 'emitter_labels': {}}

    if_layout = 'lo' in opt.cfg.DATA.data_read_list
    if_object = 'ob' in opt.cfg.DATA.data_read_list
    if_mesh = 'mesh' in opt.cfg.DATA.data_read_list
    if_emitter = 'em' in opt.cfg.DATA.data_read_list
    emitter_est_type = opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type

    if if_layout:
        pitch_reg = data_batch['camera']['pitch_reg'].float().cuda(non_blocking=True)
        pitch_cls = data_batch['camera']['pitch_cls'].long().cuda(non_blocking=True)
        roll_reg = data_batch['camera']['roll_reg'].float().cuda(non_blocking=True)
        roll_cls = data_batch['camera']['roll_cls'].long().cuda(non_blocking=True)
        if opt.cfg.MODEL_LAYOUT_EMITTER.layout.if_train_with_reindexed:
            lo_ori_reg = data_batch['layout_reindexed']['ori_reg'].float().cuda(non_blocking=True)
            lo_ori_cls = data_batch['layout_reindexed']['ori_cls'].long().cuda(non_blocking=True)
            lo_centroid = data_batch['layout_reindexed']['centroid_reg'].float().cuda(non_blocking=True)
            lo_coeffs = data_batch['layout_reindexed']['coeffs_reg'].float().cuda(non_blocking=True)
            lo_bdb3D = data_batch['layout_reindexed']['bdb3D'].float().cuda(non_blocking=True)
            lo_bdb3D_reindexed = lo_bdb3D
            lo_bdb3D_full = data_batch['layout_reindexed']
        else:
            lo_ori_reg = data_batch['layout_']['ori_reg'].float().cuda(non_blocking=True)
            lo_ori_cls = data_batch['layout_']['ori_cls'].long().cuda(non_blocking=True)
            lo_centroid = data_batch['layout_']['centroid_reg'].float().cuda(non_blocking=True)
            lo_coeffs = data_batch['layout_']['coeffs_reg'].float().cuda(non_blocking=True)
            lo_bdb3D = data_batch['layout_']['bdb3D'].float().cuda(non_blocking=True)
            lo_bdb3D_reindexed = data_batch['layout_reindexed']['bdb3D'].float().cuda(non_blocking=True)
            lo_bdb3D_full = data_batch['layout_']

        # cam_K = data_batch['camera']['K'].float().cuda(non_blocking=True)
        cam_K_scaled = data_batch['camera']['K_scaled'].float().cuda(non_blocking=True)

        cam_R_gt = get_rotation_matrix_gt(opt.bins_tensor, 
                                            pitch_cls, pitch_reg, 
                                            roll_cls, roll_reg)

        layout_labels = {'pitch_reg': pitch_reg, 'pitch_cls': pitch_cls, 'roll_reg': roll_reg,
                        'roll_cls': roll_cls, 'lo_ori_reg': lo_ori_reg, 'lo_ori_cls': lo_ori_cls, 'lo_centroid': lo_centroid,
                        'lo_coeffs': lo_coeffs, 'lo_bdb3D': lo_bdb3D, 'lo_bdb3D_reindexed': lo_bdb3D_reindexed, 'lo_bdb3D_full': lo_bdb3D_full, 'cam_K_scaled': cam_K_scaled, 'cam_R_gt':cam_R_gt}
        labels_dict['layout_labels'] = layout_labels

    if if_object:
        patch = data_batch['boxes_batch']['patch'].cuda(non_blocking=True)
        g_features = data_batch['boxes_batch']['g_feature'].float().cuda(non_blocking=True)
        size_reg = data_batch['boxes_batch']['size_reg'].float().cuda(non_blocking=True)
        size_cls = data_batch['boxes_batch']['size_cls'].float().cuda(non_blocking=True) # one hot vector of shape [n_objects, n_classes]
        ori_reg = data_batch['boxes_batch']['ori_reg'].float().cuda(non_blocking=True)
        ori_cls = data_batch['boxes_batch']['ori_cls'].long().cuda(non_blocking=True)
        centroid_reg = data_batch['boxes_batch']['centroid_reg'].float().cuda(non_blocking=True)
        centroid_cls = data_batch['boxes_batch']['centroid_cls'].long().cuda(non_blocking=True)
        offset_2D = data_batch['boxes_batch']['delta_2D'].float().cuda(non_blocking=True)
        split = data_batch['obj_split']
        # split of relational pairs for batch learning.
        rel_pair_counts = torch.cat([torch.tensor([0]), torch.cumsum(
            torch.pow(data_batch['obj_split'][:, 1] - data_batch['obj_split'][:, 0], 2), 0)], 0)

        ''' calculate loss from the interelationship between object and layout.'''
        bdb2D_from_3D_gt = data_batch['boxes_batch']['bdb2D_from_3D'].float().cuda(non_blocking=True)
        bdb2D_pos = data_batch['boxes_batch']['bdb2D_pos'].float().cuda(non_blocking=True)
        bdb3D = data_batch['boxes_batch']['bdb3D'].float().cuda(non_blocking=True)
        random_id = data_batch['boxes_batch']['random_id'] # list of lists
        cat_name = data_batch['boxes_batch']['cat_name'] # list of lists

        cam_K_scaled = data_batch['camera']['K_scaled'].float().cuda(non_blocking=True)

        object_labels = {'patch':patch, 'g_features':g_features, 'size_reg':size_reg, 'size_cls':size_cls,
            'ori_reg':ori_reg, 'ori_cls':ori_cls, 'centroid_reg':centroid_reg, 'centroid_cls':centroid_cls,
            'offset_2D':offset_2D, 'split':split, 'rel_pair_counts':rel_pair_counts, 'bdb2D_from_3D_gt':bdb2D_from_3D_gt, 'bdb2D_pos':bdb2D_pos, 'bdb3D':bdb3D, 
            'cam_K_scaled': cam_K_scaled}
        object_labels.update({'boxes_valid_list': data_batch['boxes_valid_list'], 'random_id': random_id, 'cat_name': cat_name})

        labels_dict['object_labels'] = object_labels

    if if_mesh:
        if opt.cfg.MODEL_LAYOUT_EMITTER.mesh.loss == 'SVRLoss':
            # print(labels_dict['object_labels']['size_cls'].shape, torch.sum(labels_dict['object_labels']['size_cls'], dim=1))
            # img = data_batch['img'].cuda(non_blocking=True)
            # cls = data_batch['cls'].float().cuda(non_blocking=True)
            mesh_points = data_batch['boxes_batch']['mesh_points'].float().cuda(non_blocking=True)
            densities = data_batch['boxes_batch']['densities'].float().cuda(non_blocking=True)
            gt_obj_path_alignedNew_normalized_list = data_batch['gt_obj_path_alignedNew_normalized_list'] # list of lists
            gt_obj_path_alignedNew_original_list = data_batch['gt_obj_path_alignedNew_original_list'] # list of lists
            # print(labels_dict['object_labels']['patch'].shape)
            # print(labels_dict['object_labels']['size_cls'].shape)
            labels_dict['mesh_labels'] = {
                # 'img': labels_dict_input['im_trainval_SDR'], 
                'img': labels_dict['object_labels']['patch'], 
                'cls': labels_dict['object_labels']['size_cls'], 
                'mesh_points': mesh_points, 
                'densities': densities,  
                'gt_obj_path_alignedNew_normalized_list': gt_obj_path_alignedNew_normalized_list, 
                'gt_obj_path_alignedNew_original_list': gt_obj_path_alignedNew_original_list
            }

            # elif opt.cfg.MODEL_LAYOUT_EMITTER.mesh.loss == 'ReconLoss':
            # for depth loss
            # depth_maps = [depth.float().to(device) for depth in data['depth']]

            depth_maps = data_batch['depth'].cuda(non_blocking=True)

            obj_masks = data_batch['boxes_batch']['mask']

            mask_status = get_mask_status(obj_masks, split)

            mask_flag = 1
            if 1 not in mask_status:
                mask_flag = 0

            # # Notice: we should conclude the NYU37 classes into pix3d (9) classes before feeding into the network.
            # cls_codes = torch.zeros([size_cls.size(0), 9]).cuda(non_blocking=True)
            # cls_codes[range(size_cls.size(0)), [NYU37_TO_PIX3D_CLS_MAPPING[cls.item()] for cls in
            #                                     torch.argmax(size_cls, dim=1)]] = 1
            # # cls_codes[range(size_cls.size(0)), [cls.item() for cls in
            # #                                     torch.argmax(size_cls, dim=1)]] = 1

            cls_codes = size_cls

            patch_for_mesh = patch[mask_status.nonzero()]
            cls_codes_for_mesh = cls_codes[mask_status.nonzero()]

            labels_dict['mesh_labels'].update({'depth_maps':depth_maps, 
                'obj_masks':obj_masks, 'mask_status':mask_status, 'mask_flag':mask_flag, 'patch_for_mesh':patch_for_mesh,
                'cls_codes_for_mesh':cls_codes_for_mesh})

    if if_emitter:
        emitter_labels = {}
        if emitter_est_type == 'wall_prob':
            # emitter_light_ratio_prob = data_batch['wall_grid_prob'].cuda(non_blocking=True)
            # emitter_labels['emitter_light_ratio_prob'] = emitter_light_ratio_prob
            raise ValueError('Not implemented!')
        elif emitter_est_type == 'cell_prob':
            # emitter_light_ratio_prob = data_batch['cell_prob_mean'].cuda(non_blocking=True)
            # emitter_labels['emitter_cls_prob'] = emitter_cls_prob
            raise ValueError('Not implemented!')
        elif emitter_est_type == 'cell_info':
            emitter_labels['cell_light_ratio'] = data_batch['cell_light_ratio'].cuda(non_blocking=True)
            emitter_labels['cell_cls'] = data_batch['cell_cls'].cuda(non_blocking=True)
            emitter_labels['cell_axis_abs'] = data_batch['cell_axis_abs'].cuda(non_blocking=True)
            emitter_labels['cell_axis_relative'] = data_batch['cell_axis_relative'].cuda(non_blocking=True)
            emitter_labels['cell_normal_outside'] = data_batch['cell_normal_outside'].cuda(non_blocking=True)
            emitter_labels['cell_intensity'] = data_batch['cell_intensity'].cuda(non_blocking=True) # actually LOG!!!
            emitter_labels['cell_lamb'] = data_batch['cell_lamb'].cuda(non_blocking=True)
        else:
            raise ValueError('Invalid: config.emitters.est_type')

        emitter_labels.update({'emitter2wall_assign_info_list': data_batch['emitter2wall_assign_info_list'], 'emitters_obj_list': data_batch['emitters_obj_list'], 'gt_layout_RAW': data_batch['gt_layout_RAW'], 'cell_info_grid': data_batch['cell_info_grid']})
        labels_dict['emitter_labels'] = emitter_labels

        if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.sample_envmap:
            emitter_labels['im_envmap_ori'] = data_batch['im_envmap_ori'].cuda(non_blocking=True)
            emitter_labels['transform_R_RAW2Total3D'] = data_batch['transform_R_RAW2Total3D'].cuda(non_blocking=True)

    return labels_dict

def vis_layout_emitter(labels_dict, output_dict, data_batch, opt, time_meters=None, batch_size_id=[]):
    batch_size = labels_dict['imBatch'].shape[0]

    batch_size_fixed, batch_id = batch_size_id[0], batch_size_id[1]
    grid_size = opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size

    output_vis_dict = {}
    if_est_emitter = 'em' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list
    if_est_layout = 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list
    if_est_object = 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list
    if_est_mesh = 'mesh' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list

    if_load_emitter = 'em' in opt.cfg.DATA.data_read_list
    if_load_layout = 'lo' in opt.cfg.DATA.data_read_list
    if_load_object = 'ob' in opt.cfg.DATA.data_read_list
    if_load_mesh = 'mesh' in opt.cfg.DATA.data_read_list
    
    if if_load_layout:
        gt_dict_lo = labels_dict['layout_labels']
    if if_load_object:
        gt_dict_ob = labels_dict['object_labels']
    if if_load_mesh:
        gt_dict_mesh = labels_dict['mesh_labels']
    
    if if_est_layout:
        pred_dict_lo = output_dict['layout_est_result']
        lo_bdb3D_out, basis_out, coeffs_out, centroid_out = get_layout_bdb_sunrgbd(opt.bins_tensor, pred_dict_lo['lo_ori_reg_result'],
                                            pred_dict_lo['lo_ori_cls_result'],
                                            pred_dict_lo['lo_centroid_result'],
                                            pred_dict_lo['lo_coeffs_result'], 
                                            if_return_full=True, 
                                            # if_differentiable=not opt.cfg.MODEL_LAYOUT_EMITTER.layout.if_argmax_in_results)
                                            if_differentiable=opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_differentiable_layout_input)
        # ic(basis_out.shape, coeffs_out.shape, centroid_out.shape) # [4, 3, 3], [4, 3], [4, 3]
        cam_R_out = get_rotation_matix_result(opt.bins_tensor,
                                    pred_dict_lo['pitch_cls_result'], pred_dict_lo['pitch_reg_result'],
                                    pred_dict_lo['roll_cls_result'], pred_dict_lo['roll_reg_result'], \
                                    # if_differentiable=not opt.cfg.MODEL_LAYOUT_EMITTER.layout.if_argmax_in_results)
                                    if_differentiable=opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_differentiable_layout_input)

    if if_est_object:
        pred_dict_ob = output_dict['object_est_result']

        # projected center
        P_result = torch.stack(((gt_dict_ob['bdb2D_pos'][:, 0] + gt_dict_ob['bdb2D_pos'][:, 2]) / 2 -
                                (gt_dict_ob['bdb2D_pos'][:, 2] - gt_dict_ob['bdb2D_pos'][:, 0]) * pred_dict_ob['offset_2D_result'][:, 0],
                                (gt_dict_ob['bdb2D_pos'][:, 1] + gt_dict_ob['bdb2D_pos'][:, 3]) / 2 -
                                (gt_dict_ob['bdb2D_pos'][:, 3] - gt_dict_ob['bdb2D_pos'][:, 1]) * pred_dict_ob['offset_2D_result'][:,1]), 1)

        cam_R_use = gt_dict_lo['cam_R_gt'] if not if_est_layout else cam_R_out
        bdb3D_out_form_cpu, bdb3D_out = get_bdb_evaluation(opt.bins_tensor,
                                                        torch.argmax(pred_dict_ob['ori_cls_result'], 1),
                                                        pred_dict_ob['ori_reg_result'],
                                                        torch.argmax(pred_dict_ob['centroid_cls_result'], 1),
                                                        pred_dict_ob['centroid_reg_result'],
                                                        gt_dict_ob['size_cls'], pred_dict_ob['size_reg_result'], P_result,
                                                        gt_dict_ob['cam_K_scaled'], cam_R_use, gt_dict_ob['split'], return_bdb=True)

    if if_est_mesh:
        pred_dict_mesh = output_dict['mesh_est_result']
        mesh_output = pred_dict_mesh['mesh_coordinates_results'][-1] # models/total3d/modules/network.py, L134
        # mesh_output = mesh_output[-1]
        # convert to SUNRGBD coordinates
        # print(mesh_output.shape)
        # mesh_output[:, 2, :] *= -1 # negate z axis
        out_faces = pred_dict_mesh['faces']

    if if_est_emitter:
        if_lightAccu = False
        gt_dict_em = labels_dict['emitter_labels']
        pred_dict_em = output_dict['emitter_est_result']
        emitter_cls_result = pred_dict_em['cell_light_ratio'].view((batch_size, 6, -1))
        if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss_type == 'KL':
            emitter_cls_result_postprocessed = torch.nn.functional.softmax(emitter_cls_result, dim=2)
        elif opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss_type == 'L2':
            emitter_cls_result_postprocessed = emitter_cls_result
            assert not(opt.cfg.MODEL_LAYOUT_EMITTER.emitter.sigmoid and opt.cfg.MODEL_LAYOUT_EMITTER.emitter.softmax), 'softmax and sigmoid cannot be True at the same time!'
            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.sigmoid:
                emitter_cls_result_postprocessed = torch.sigmoid(emitter_cls_result)
            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.softmax:
                emitter_cls_result_postprocessed = torch.nn.functional.softmax(emitter_cls_result, dim=2)
        
        if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
            cell_cls = torch.argmax(pred_dict_em['cell_cls'].view((batch_size, 6, grid_size, grid_size, 3)), -1).detach().cpu().numpy()
            cell_axis = pred_dict_em['cell_axis'].view((batch_size, 6, grid_size, grid_size, 3)).detach().cpu().numpy()
            cell_intensity = pred_dict_em['cell_intensity'].view((batch_size, 6, grid_size, grid_size, 3)).detach().cpu().numpy()
            cell_lamb = pred_dict_em['cell_lamb'].view((batch_size, 6, grid_size, grid_size)).detach().cpu().numpy()
            if 'emitter_outdirs_meshgrid_Total3D_outside_abs' in pred_dict_em:
                if_lightAccu = True
                envHeight = opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.envHeight
                envWidth = opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.envWidth
                emitter_outdirs_meshgrid_Total3D_outside_abs = pred_dict_em['emitter_outdirs_meshgrid_Total3D_outside_abs'].detach().cpu().numpy().reshape(batch_size, 6, grid_size, grid_size, envHeight, envWidth, 3)
                normal_outside_Total3D = pred_dict_em['normal_outside_Total3D'].detach().cpu().numpy().reshape(batch_size, 6, grid_size, grid_size, 3)


    scene_box_list = []
    layout_info_dict_list = []
    emitter_info_dict_list = []

    # print(gt_dict_ob['split'])
    # print(gt_dict_ob['boxes_valid_list'])
    for sample_idx_batch in range(batch_size):
        sample_idx = sample_idx_batch+batch_size_fixed*batch_id
        # print('--- Visualizing sample %d ---'%sample_idx_batch)
        # save_prefix = 'sample%d-LABEL-epoch%d-tid%d-%s'%(sample_idx_batch+batch_size*vis_batch_count, epoch, iter, phase)
        gt_cam_R = gt_dict_lo['cam_R_gt'][sample_idx_batch].cpu().numpy()
        cam_K = gt_dict_lo['cam_K_scaled'][sample_idx_batch].cpu().numpy()
        # gt_layout = gt_dict_lo['lo_bdb3D'][sample_idx_batch].cpu().numpy()
        if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_train_with_reindexed_layout:
            gt_layout = gt_dict_lo['lo_bdb3D_reindexed'][sample_idx_batch].cpu().numpy()
        else:
            gt_layout = gt_dict_lo['lo_bdb3D'][sample_idx_batch].cpu().numpy()

        gt_layout_full = {key: gt_dict_lo['lo_bdb3D_full'][key][sample_idx_batch].detach().cpu().numpy() for key in gt_dict_lo['lo_bdb3D_full']}

        # ---- layout
        if if_est_layout:
            pre_layout = lo_bdb3D_out[sample_idx_batch, :, :].cpu().detach().numpy()
            pre_layout_full = {'bdb3D': pre_layout, 'coeffs': coeffs_out[sample_idx_batch].cpu().detach().numpy(), 'basis': basis_out[sample_idx_batch].cpu().detach().numpy(), 'centroid': centroid_out[sample_idx_batch].cpu().detach().numpy()}
            pre_cam_R = cam_R_out[sample_idx_batch, :, :].cpu().detach().numpy()
            if opt.cfg.MODEL_LAYOUT_EMITTER.layout.if_train_with_reindexed:
                pre_layout = reindex_layout(pre_layout, pre_cam_R)
                pre_layout_full['bdb3D'] = pre_layout
            
        else:
            pre_cam_R = gt_cam_R
            pre_layout = gt_layout
            pre_layout_full = gt_layout_full

        # ---- objects
        if if_load_object or if_load_mesh:
            interval = gt_dict_ob['split'][sample_idx_batch].cpu().tolist()

        if if_load_object:
            if sum(gt_dict_ob['boxes_valid_list'][sample_idx_batch])==0: # if there are no objects
                gt_boxes=None
            else:
                gt_boxes = format_bboxes({'bdb3D': gt_dict_ob['bdb3D'][interval[0]:interval[1]].cpu().numpy(), 
                    'class_id': gt_dict_ob['size_cls'][interval[0]:interval[1]].cpu().argmax(1).flatten().numpy().tolist()}, 'GT')
                gt_boxes['bdb2d'] = gt_dict_ob['bdb2D_pos'][interval[0]:interval[1]].cpu().numpy()
                gt_boxes['random_id'] = gt_dict_ob['random_id'][sample_idx_batch]
                gt_boxes['cat_name'] = gt_dict_ob['cat_name'][sample_idx_batch]
                gt_boxes['if_valid'] = gt_dict_ob['boxes_valid_list'][sample_idx_batch]
                assert len(gt_boxes['if_valid']) == len(gt_boxes['random_id'])== len(gt_boxes['cat_name'])
                # print(len(gt_dict_ob['bdb3D']), data_batch['obj_split'])
                # print(len(gt_dict_mesh['obj_masks']), len(gt_dict_mesh['obj_masks'][0]), gt_dict_mesh['obj_masks'][0][0].keys())
                if if_load_mesh:
                    obj_masks = gt_dict_mesh['obj_masks'][sample_idx_batch]
                    assert len(gt_boxes['if_valid'])==len(obj_masks)
                    gt_boxes['obj_masks'] = obj_masks
        else:
            gt_boxes = None


        if if_est_object:
            nyu40class_ids = [int(evaluate_bdb['classid']) for evaluate_bdb in bdb3D_out_form_cpu]
            # save bounding boxes and camera poses
            current_cls = nyu40class_ids[interval[0]:interval[1]]

            # bdb3d_mat_path = os.path.join(str(save_path), '%s_bdb_3d.mat'%save_prefix)
            pre_box_data = {'bdb': bdb3D_out_form_cpu[interval[0]:interval[1]], 'class_id': current_cls}

            # class_ids = gt_dict_ob['size_cls'][interval[0]:interval[1]].cpu().argmax(1).flatten().numpy().tolist()
            if sum(gt_dict_ob['boxes_valid_list'][sample_idx_batch])==0:
                pre_boxes = None
            else:
                pre_boxes = format_bboxes(pre_box_data, 'prediction')
                pre_boxes['if_valid'] = gt_boxes['if_valid']
                pre_boxes['random_id'] = gt_boxes['random_id']
                pre_boxes['cat_name'] = gt_boxes['cat_name']
        else:
            pre_boxes = gt_boxes

        # ---- meshes
        if if_est_mesh:
            current_faces = out_faces[interval[0]:interval[1]].cpu().numpy()
            current_coordinates = mesh_output.transpose(1, 2)[interval[0]:interval[1]].cpu().numpy()
            pre_meshes = []

            # print(sample_idx_batch, current_faces.shape, current_coordinates.shape, interval)
            num_objs = interval[1] - interval[0]
            
            for obj_id in range(num_objs):
                save_path = Path(opt.summary_vis_path_task) / ('sample%d-obj%s.obj' % (sample_idx, obj_id))
                mesh_obj = {'v': current_coordinates[obj_id],
                            'f': current_faces[obj_id]}
                write_obj(save_path, mesh_obj)
                pre_meshes.append([save_path, mesh_obj])

            # gt_meshes = gt_dict_mesh['gt_obj_path_alignedNew_normalized_list'][sample_idx_batch]
            gt_meshes = gt_dict_mesh['gt_obj_path_alignedNew_original_list'][sample_idx_batch]
        else:
            gt_meshes = None
            pre_meshes = None

        image = (labels_dict['im_fixedscale_SDR'][sample_idx_batch].detach().cpu().numpy() * 255.).astype(np.uint8)
        # image = np.transpose(image, (1, 2, 0))

        # ---- emitters
        if if_load_emitter:
            emitters_obj_list_gt = gt_dict_em['emitters_obj_list'][sample_idx_batch]
            emitter2wall_assign_info_list_gt = gt_dict_em['emitter2wall_assign_info_list'][sample_idx_batch]
            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
                emitter_cls_prob_GT_np = gt_dict_em['cell_light_ratio'][sample_idx_batch].detach().cpu().numpy()
            else:
                emitter_cls_prob_GT_np = gt_dict_em['emitter_cls_prob'][sample_idx_batch].detach().cpu().numpy()

            assert opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info'
            cell_info_grid_GT_includeempty = gt_dict_em['cell_info_grid'][sample_idx_batch]
            cell_info_grid_GT = []
            for wall_idx in range(6):
                for i in range(grid_size):
                    for j in range(grid_size):
                        cell_info = cell_info_grid_GT_includeempty[wall_idx * grid_size**2 + i * grid_size + j]
                        if cell_info['obj_type'] is not None:
                            cell_info['wallidx_i_j'] = (wall_idx, i, j)
                            cell_info_grid_GT.append(cell_info)
            cell_normal_outside_gt_np = gt_dict_em['cell_normal_outside'][sample_idx_batch].detach().cpu().numpy() # [6, 8, 8, 3]
        else:
            emitters_obj_list_gt = None
            emitter2wall_assign_info_list_gt = None
            emitter_cls_prob_GT_np = None
            cell_info_grid_GT = None
            cell_normal_outside_gt_np = None

        if if_est_emitter:
            # gt_layout_RAW = gt_dict_em['gt_layout_RAW'][sample_idx_batch]
            emitter_cls_result_postprocessed_np = emitter_cls_result_postprocessed[sample_idx_batch].detach().cpu().numpy() # [102]

            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
                if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_train_with_reindexed_layout:
                    face_v_idxes = [(1, 0, 2, 3), (4, 5, 7, 6), (0, 1, 4, 5), (1, 2, 5, 6), (3, 7, 2, 6), (4, 7, 0, 3)]

                map_obj_type_int = {1: 'window', 2: 'obj', 0: 'null'}
                cell_info_grid_PRED = []
                for wall_idx in range(6):
                    # if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_train_with_reindexed_layout:
                    #     basis_index = face_v_idxes[wall_idx]
                    #     normal_outside = - np.cross(gt_layout_RAW[basis_index[1]] - gt_layout_RAW[basis_index[0]], gt_layout_RAW[basis_index[2]] - gt_layout_RAW[basis_index[0]]).flatten() # [TODO] change to use EST layout when joint training!!!
                    #     normal_outside = normal_outside / np.linalg.norm(normal_outside)
                    for i in range(grid_size):
                        for j in range(grid_size):
                            if not opt.cfg.MODEL_LAYOUT_EMITTER.emitter.cls_agnostric:
                                if cell_cls[sample_idx_batch][wall_idx][i][j] == 0:
                                    continue
                            # if not opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_train_with_reindexed_layout:
                            #     normal_outside = cell_normal_outside_gt_np[wall_idx, i, j]

                            cell_info = {'obj_type': map_obj_type_int[cell_cls[sample_idx_batch][wall_idx][i][j]], 'emitter_info': {}, 'wallidx_i_j': (wall_idx, i, j)}
                            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.relative_dir:
                                cell_info['emitter_info']['light_dir_offset'] = cell_axis[sample_idx_batch][wall_idx][i][j].flatten()
                                # cell_info['emitter_info']['light_dir_abs'] = cell_info['emitter_info']['light_dir_offset'] + normal_outside
                            else:
                                cell_info['emitter_info']['light_dir_abs'] = cell_axis[sample_idx_batch][wall_idx][i][j].flatten()

                            # cell_info['emitter_info']['normal_outside'] = normal_outside

                            if if_lightAccu:
                                emitter_outdirs_meshgrid_Total3D_outside_abs_single = emitter_outdirs_meshgrid_Total3D_outside_abs[sample_idx_batch, wall_idx, i, j]
                                normal_outside_Total3D_single = normal_outside_Total3D[sample_idx_batch, wall_idx, i, j]
                                # if not opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_use_est_layout:
                                #     assert np.amax(np.abs(normal_outside_Total3D_single - normal_outside)) < 1e-3
                                # print(wall_idx, i, j, normal_outside_Total3D_single, normal_outside)
                                cell_info['emitter_info']['emitter_outdirs_meshgrid_Total3D_outside_abs'] = emitter_outdirs_meshgrid_Total3D_outside_abs_single # [8, 16, 3]
                                cell_info['emitter_info']['normal_outside_Total3D_single'] = normal_outside_Total3D_single
                            
                            intensity_log = cell_intensity[sample_idx_batch][wall_idx][i][j].flatten() # actually predicts LOG intensity!
                            assert intensity_log.shape == (3,)
                            intensity = np.exp(intensity_log) - 1.
                            intensity_scale255 = np.amax(intensity) / 255.
                            intensity_scaled255 = intensity / (intensity_scale255 + 1e-5)
                            intensity_scaled01 = [np.clip(x/255., 0., 1.) for x in intensity_scaled255] # max: 1.
                            intensity_scalelog = np.log(np.clip(np.linalg.norm(intensity.flatten()) + 1., 1., np.inf))
                            cell_info['emitter_info']['intensity_scalelog'] = intensity_scalelog # log of norm of intensity
                            cell_info['emitter_info']['intensity_scaled01'] = intensity_scaled01
                            cell_info['emitter_info']['intensity'] = intensity  
                            cell_info['emitter_info']['lamb'] = cell_lamb[sample_idx_batch][wall_idx][i][j].item()
                            cell_info['light_ratio'] = emitter_cls_result_postprocessed_np[wall_idx][i * grid_size + j]
                            cell_info_grid_PRED.append(cell_info)
            else:
                cell_info_grid_PRED = None
        else:
            # gt_layout_RAW = None
            emitter_cls_result_postprocessed_np = None
            cell_info_grid_PRED = None

        save_prefix = ''
        
        if opt.cfg.DATA.load_layout_emitter_gt:
            transform_R = data_batch['transform_R_RAW2Total3D'][sample_idx_batch].cpu().numpy().reshape(3, 3)
            transform_t = data_batch['transform_t_RAW2Total3D'][sample_idx_batch].cpu().numpy().reshape(3, 1)
            
        if if_load_emitter:
            hdr_scale = data_batch['hdr_scale'][sample_idx_batch].cpu().numpy().item()
            env_scale = data_batch['env_scale'][sample_idx_batch].cpu().numpy().item()
        else:
            hdr_scale = 1.
            env_scale = 1.
        scene_box = Box(image, cam_K, gt_cam_R, pre_cam_R, gt_layout_full, pre_layout_full, gt_boxes, pre_boxes, gt_meshes, pre_meshes, \
            opt=opt, dataset='OR', description=save_prefix, if_mute_print=True, OR=opt.cfg.MODEL_LAYOUT_EMITTER.data.OR, \
            emitter2wall_assign_info_list_gt = emitter2wall_assign_info_list_gt, 
            emitters_obj_list_gt = emitters_obj_list_gt, 
            # gt_layout_RAW = gt_layout_RAW, 
            emitter_cls_prob_PRED = emitter_cls_result_postprocessed_np, 
            emitter_cls_prob_GT = emitter_cls_prob_GT_np, 
            cell_info_grid_GT = cell_info_grid_GT, 
            cell_info_grid_PRED = cell_info_grid_PRED, 
            grid_size = grid_size,
            if_use_vtk = opt.cfg.MODEL_LAYOUT_EMITTER.mesh.if_use_vtk, 
            transform_R=transform_R, transform_t=transform_t, hdr_scale=hdr_scale, env_scale=env_scale)

        scene_box_list.append(scene_box)
        layout_info_dict_list.append({'est_data': None, 'gt_cam_R': gt_cam_R, 'cam_K': cam_K, \
            'gt_layout': gt_layout, 'pre_layout': pre_layout, 'gt_layout_full': gt_layout_full, 'pre_layout_full': pre_layout_full, \
            'gt_meshes': gt_meshes, 'pre_meshes': pre_meshes, 'gt_boxes': gt_boxes, 'pre_boxes': pre_boxes, \
            'pre_cam_R': pre_cam_R, 'image': image})
        emitter_info_dict = {'cell_info_grid_GT': cell_info_grid_GT, 'cell_info_grid_PRED': cell_info_grid_PRED, \
                                'emitter_cls_prob_PRED': emitter_cls_result_postprocessed_np, 'emitter_cls_prob_GT': emitter_cls_prob_GT_np, \
                                'pre_layout': pre_layout, 'pre_cam_R': pre_cam_R}
        if if_est_emitter and opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.enable:
            envmap_lightAccu_mean = pred_dict_em['envmap_lightAccu_mean'][sample_idx_batch].detach().cpu().numpy()
            # print(envmap_lightAccu_mean.shape) # (6, 3, 8, 8)
            envmap_lightAccu_mean_vis = envmap_lightAccu_mean**(1.0/2.2)

            emitter_info_dict.update({'envmap_lightAccu_mean_vis_GT': envmap_lightAccu_mean_vis})

        emitter_info_dict_list.append(emitter_info_dict)

    output_vis_dict['scene_box_list'] = scene_box_list
    output_vis_dict['layout_info_dict_list'] = layout_info_dict_list
    output_vis_dict['emitter_info_dict_list'] = emitter_info_dict_list

    # output_vis_dict.update({
    #     'cell_info_grid_GT': cell_info_grid_GT, \
    #     'cell_info_grid_PRED': cell_info_grid_PRED, \
    #     'emitter_cls_prob_PRED': emitter_cls_result_postprocessed_np, 'emitter_cls_prob_GT': emitter_cls_prob_GT_np, \
    #     'pre_layout': pre_layout, 'pre_cam_R': pre_cam_R
    # })


    return output_vis_dict


def postprocess_layout_object_emitter(labels_dict, output_dict, loss_dict, opt, time_meters, is_train, if_vis=False, ):
    if 'ob' in opt.cfg.DATA.data_read_list or 'mesh' in opt.cfg.DATA.data_read_list:
        flattened_valid_mask = [item for sublist in labels_dict['object_labels']['boxes_valid_list'] for item in sublist]
        flattened_valid_mask_tensor = torch.tensor(flattened_valid_mask).float().cuda()

    if 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
        output_dict, loss_dict = postprocess_layout(labels_dict, output_dict, loss_dict, opt, time_meters)
    if 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
        output_dict, loss_dict = postprocess_object(labels_dict, output_dict, loss_dict, opt, time_meters, flattened_valid_mask_tensor=flattened_valid_mask_tensor)
    if 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list and 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
        output_dict, loss_dict = postprocess_joint(labels_dict, output_dict, loss_dict, opt, time_meters, flattened_valid_mask_tensor=flattened_valid_mask_tensor)
    if 'mesh' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
        output_dict, loss_dict = postprocess_mesh(labels_dict, output_dict, loss_dict, opt, time_meters, is_train=is_train, flattened_valid_mask_tensor=flattened_valid_mask_tensor)

    if 'em' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
        # input_dict_extra = {}
        # if 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list and opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_use_est_layout:
        #     input_dict_extra.update('lo_bdb3D_result': output_dict['results_layout']['lo_bdb3D_result']})
        output_dict, loss_dict = postprocess_emitter(labels_dict, output_dict, loss_dict, opt, time_meters)

    return output_dict, loss_dict

def postprocess_layout(labels_dict, output_dict, loss_dict, opt, time_meters):
    cls_reg_ratio = opt.cfg.MODEL_LAYOUT_EMITTER.layout.loss.cls_reg_ratio
    pred_dict = output_dict['layout_est_result']
    gt_dict = labels_dict['layout_labels']

    layout_losses_dict, lo_bdb3D_result = PoseLoss(opt, pred_dict, gt_dict, opt.bins_tensor, cls_reg_ratio)

    loss_dict.update(layout_losses_dict)
    loss_dict.update({'loss_layout-ALL': sum([layout_losses_dict[x] for x in layout_losses_dict])})
    # print('=======', [x for x in layout_losses_dict if 'cls' not in x])
    # loss_dict.update({'loss_layout-ALL': sum([layout_losses_dict[x] if 'cls' not in x else torch.zeros(1).cuda() for x in layout_losses_dict])})

    output_dict.update({'results_layout': {'lo_bdb3D_result': lo_bdb3D_result}})

    return output_dict, loss_dict

def postprocess_object(labels_dict, output_dict, loss_dict, opt, time_meters, flattened_valid_mask_tensor=None):
    cls_reg_ratio = opt.cfg.MODEL_LAYOUT_EMITTER.layout.loss.cls_reg_ratio
    pred_dict = output_dict['object_est_result']
    gt_dict = labels_dict['object_labels']

    object_losses_dict = DetLoss(pred_dict, gt_dict, cls_reg_ratio, flattened_valid_mask_tensor=flattened_valid_mask_tensor)

    loss_dict.update(object_losses_dict)
    loss_dict.update({'loss_object-ALL': sum([object_losses_dict[x] for x in object_losses_dict])})

    return output_dict, loss_dict

def postprocess_joint(labels_dict, output_dict, loss_dict, opt, time_meters, flattened_valid_mask_tensor=None):
    cls_reg_ratio = opt.cfg.MODEL_LAYOUT_EMITTER.layout.loss.cls_reg_ratio
    pred_dict = {**output_dict['object_est_result'], **output_dict['layout_est_result']}
    gt_dict = {**labels_dict['object_labels'], **labels_dict['layout_labels']}

    joint_losses_dict, joint_outputs_dict = JointLoss(pred_dict, gt_dict, opt.bins_tensor, output_dict['results_layout']['lo_bdb3D_result'], cls_reg_ratio, flattened_valid_mask_tensor=flattened_valid_mask_tensor)

    loss_dict.update(joint_losses_dict)
    loss_dict.update({'loss_joint-ALL': sum([joint_losses_dict[x] for x in joint_losses_dict])})

    output_dict['results_joint'] = joint_outputs_dict

    return output_dict, loss_dict

def postprocess_mesh(labels_dict, output_dict, loss_dict, opt, time_meters, is_train, flattened_valid_mask_tensor=None):
    pred_dict = output_dict['mesh_est_result']
    gt_dict = labels_dict['mesh_labels']

    if opt.cfg.MODEL_LAYOUT_EMITTER.mesh.loss == 'SVRLoss':
        mesh_losses_dict = SVRLoss(pred_dict, gt_dict, 
            subnetworks=opt.cfg.MODEL_LAYOUT_EMITTER.mesh.tmn_subnetworks, face_sampling_rate=opt.cfg.MODEL_LAYOUT_EMITTER.mesh.face_samples, 
            flattened_valid_mask_tensor=flattened_valid_mask_tensor, is_train=is_train)
    elif opt.cfg.MODEL_LAYOUT_EMITTER.mesh.loss == 'ReconLoss':
        mesh_losses_dict = ReconLoss(pred_dict, gt_dict, extra_results=output_dict['results_joint'])

    loss_dict.update(mesh_losses_dict)
    loss_dict.update({'loss_mesh-ALL': sum([mesh_losses_dict[x] for x in mesh_losses_dict])})

    return output_dict, loss_dict


def postprocess_emitter(labels_dict, output_dict, loss_dict, opt, time_meters, input_dict_extra={}):
    
    loss_type = opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss_type
    assert loss_type in ['L2', 'KL']
    est_type = opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type

    emitter_light_ratio_est = output_dict['emitter_est_result']['cell_light_ratio'] # torch.Size([9, 102])
    B = emitter_light_ratio_est.shape[0]
    emitter_light_ratio_est = emitter_light_ratio_est.view((B, 6, -1))
    if est_type == 'cell_info':
        emitter_light_ratio_gt = labels_dict['emitter_labels']['cell_light_ratio'].view((B, 6, -1))
    else:
        emitter_light_ratio_gt = labels_dict['emitter_labels']['emitter_light_ratio_prob'].view((B, 6, -1))

    valid_mask = (labels_dict['emitter_labels']['cell_cls'] != 0).float().view((B, 6, -1))
    window_mask = (labels_dict['emitter_labels']['cell_cls'] == 1).float().view((B, 6, -1))

    if loss_type == 'KL':
        assert est_type == 'wall_prob', 'KL loss can only be used with wall_prob where all cells of a wall sum up to 1.'
        emitter_light_ratio_est = torch.nn.functional.softmax(emitter_light_ratio_est, dim=2)
        emitter_light_ratio_loss = emitter_cls_criterion_mean(emitter_light_ratio_est, emitter_light_ratio_gt)
    elif loss_type == 'L2':
        assert not(opt.cfg.MODEL_LAYOUT_EMITTER.emitter.sigmoid and opt.cfg.MODEL_LAYOUT_EMITTER.emitter.softmax), 'softmax and sigmoid cannot be True at the same time!'
        if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.sigmoid:
            emitter_light_ratio_est = torch.sigmoid(emitter_light_ratio_est)
        if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.softmax:
            emitter_light_ratio_est = torch.nn.functional.softmax(emitter_light_ratio_est, dim=2)
        # emitter_light_ratio_loss = emitter_cls_criterion_L2_mean(emitter_light_ratio_est, emitter_light_ratio_gt)
        # print(emitter_light_ratio_est[valid_mask!=0.].detach().cpu().numpy())
        # print(emitter_light_ratio_gt[valid_mask!=0.].detach().cpu().numpy())
        # print(valid_mask.shape)
        emitter_light_ratio_loss = emitter_cls_criterion_L2_none(emitter_light_ratio_est, emitter_light_ratio_gt)
        emitter_light_ratio_loss = torch.sum(emitter_light_ratio_loss * valid_mask) / (torch.sum(valid_mask) + 1e-5)
        # print('------', emitter_light_ratio_loss*opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss.weight_light_ratio, emitter_light_ratio_loss)

    else:
        raise ValueError('Unrecognized emitter cls loss type: ' + loss_type)
    
    emitter_light_ratio_loss = emitter_light_ratio_loss * opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss.weight_light_ratio
    emitter_losses_dict, emitter_results_dict = {'loss_emitter-light_ratio': emitter_light_ratio_loss}, {'emitter_light_ratio_est_from_loss': emitter_light_ratio_est}

    if est_type == 'cell_info':

        for head_name, head_channels in [('cell_cls', 3), ('cell_axis', 3), ('cell_intensity', 3), ('cell_lamb', 1)]:
            emitter_fc_output = output_dict['emitter_est_result'][head_name] # torch.Size([9, 102])
            head_name_to_label_name_dict = {'cell_cls': 'cell_cls', 'cell_axis': 'cell_axis_abs', 'cell_intensity': 'cell_intensity', 'cell_lamb': 'cell_lamb'}
            label_name = head_name_to_label_name_dict[head_name]
            if head_name == 'cell_cls':
                emitter_fc_output = emitter_fc_output.view((B, 6, -1, 3)).permute(0, 3, 1, 2)
                emitter_property_gt = labels_dict['emitter_labels'][label_name].view((B, 6, -1))
            elif head_name in ['cell_axis', 'cell_intensity']:
                emitter_fc_output = emitter_fc_output.view((B, 6, -1, 3))
                emitter_property_gt = labels_dict['emitter_labels'][label_name].view((B, 6, -1, 3))

            elif head_name in ['cell_lamb']:
                emitter_fc_output = emitter_fc_output.view((B, 6, -1))
                emitter_property_gt = labels_dict['emitter_labels'][label_name].view((B, 6, -1))


            if head_name == 'cell_cls':
                loss = cls_criterion_mean(emitter_fc_output, emitter_property_gt)
                loss = loss * opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_cls
            elif head_name == 'cell_axis':
                # print(emitter_fc_output.shape, emitter_property_gt.shape, window_mask.shape) # torch.Size([2, 6, 64, 3]) torch.Size([2, 6, 64, 3], torch.Size([2, 6, 64])
                # # print(torch.norm(emitter_fc_output, dim=-1)[window_mask==1.].flatten())
                # # print(torch.norm(emitter_property_gt, dim=-1)[window_mask==1.].flatten())
                # emitter_fc_output, emitter_property_gt

                # window_mask_expand = window_mask.unsqueeze(-1)
                # emitter_property_gt = window_mask_expand * emitter_property_gt
                # emitter_fc_output_scaled = LSregress(emitter_fc_output * window_mask_expand.expand_as(emitter_fc_output ),
                #         emitter_property_gt * window_mask_expand.expand_as(emitter_property_gt), emitter_fc_output )
                # assert emitter_fc_output_scaled.shape == emitter_fc_output.shape

                # print(torch.norm(emitter_fc_output, dim=-1)[window_mask==1.])
                # print(torch.norm(emitter_fc_output_scaled, dim=-1)[window_mask==1.])
                # print(torch.norm(emitter_property_gt, dim=-1)[window_mask==1.])

                if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.relative_dir:
                    cell_normal_outside = labels_dict['emitter_labels']['cell_normal_outside'].view(B, 6, -1, 3)
                    # print(cell_normal_outside.shape, emitter_fc_output.shape, emitter_property_gt.shape)
                    emitter_cell_axis_abs_est = emitter_fc_output + cell_normal_outside
                else:
                    # abs_dir label was not normalized
                    emitter_cell_axis_abs_est = emitter_fc_output
                    if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.scale_invariant_loss_for_cell_axis:
                        emitter_property_gt_norm = torch.linalg.norm(emitter_property_gt, dim=-1, keepdim=True)
                        emitter_property_gt = emitter_property_gt / (emitter_property_gt_norm+1e-6)

                # print(cell_normal_outside.shape, window_mask.shape) # torch.Size([2, 6, 64, 3]) torch.Size([2, 6, 64])
                # a = cell_normal_outside[window_mask==1.]
                # a = a.reshape(-1, 3)
                # print(a.shape, a, '---456---')

                if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.scale_invariant_loss_for_cell_axis:
                    # emitter_fc_output_norm = torch.linalg.norm(emitter_cell_axis_abs_est, dim=-1, keepdim=True).detach()
                    emitter_fc_output_norm = torch.linalg.norm(emitter_cell_axis_abs_est, dim=-1, keepdim=True)
                    emitter_cell_axis_abs_est = emitter_cell_axis_abs_est / (emitter_fc_output_norm+1e-6)

                loss = emitter_cls_criterion_L2_none(emitter_cell_axis_abs_est, emitter_property_gt)
                loss = torch.sum(loss * window_mask.unsqueeze(-1)) / (torch.sum(window_mask.unsqueeze(-1)) * 3. + 1e-5) # only care the axis of windows; lamps are modeled as omnidirectional
                loss = loss * opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_axis_global
                emitter_results_dict.update({'emitter_cell_axis_abs_est': emitter_cell_axis_abs_est, 'emitter_cell_axis_abs_gt': emitter_property_gt, 'window_mask': window_mask})
            else:
                loss = emitter_cls_criterion_L2_none(emitter_fc_output, emitter_property_gt)
                if head_name == 'cell_intensity':
                    loss = torch.sum(loss * valid_mask.unsqueeze(-1)) / (torch.sum(valid_mask.unsqueeze(-1)) * 3. + 1e-5)
                    loss = loss * opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_intensity
                    # print(emitter_fc_output.shape, emitter_property_gt.shape, valid_mask.unsqueeze(-1).shape)
                    # print('-', emitter_fc_output[valid_mask==1])
                    # print('----', emitter_property_gt[valid_mask==1])
                    # print('=', torch.exp(emitter_fc_output[valid_mask==1])-1.)
                    # print('===', torch.exp(emitter_property_gt[valid_mask==1])-1.)
                elif head_name == 'cell_lamb': # lamps have no lambda
                    loss = torch.sum(loss * window_mask) / (torch.sum(window_mask) + 1e-5)
                    loss = loss * opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_lamb


            emitter_results_dict.update({head_name+'_result_from_loss': emitter_fc_output})

            # if head_name == 'cell_axis':
            #     emitter_losses_dict.update({'loss_emitter-'+head_name: torch.zeros(1).cuda()})
            #     continue

            emitter_losses_dict.update({'loss_emitter-'+head_name: loss})

    loss_dict.update(emitter_losses_dict)
    loss_dict.update({'loss_emitter-ALL': sum([emitter_losses_dict[x] for x in emitter_losses_dict])})
    output_dict.update({'results_emitter': emitter_results_dict})
    return output_dict, loss_dict