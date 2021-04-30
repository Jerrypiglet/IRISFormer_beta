import torch
import torch.nn as nn
from torch.autograd import Variable
# import models
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import statistics
import torchvision.utils as vutils
from utils.utils_total3D.utils_OR_cam import get_rotation_matrix_gt, get_rotation_matix_result
from utils.utils_total3D.utils_OR_layout import get_layout_bdb_sunrgbd, reindex_layout
from utils.utils_total3D.utils_OR_visualize import Box

from models_def.models_brdf import LSregress

cls_criterion = nn.CrossEntropyLoss(reduction='mean')
reg_criterion = nn.SmoothL1Loss(reduction='mean')
mse_criterion = nn.MSELoss(reduction='mean')
binary_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')

cls_criterion_none = nn.CrossEntropyLoss(reduction='none')
cls_criterion_mean = nn.CrossEntropyLoss(reduction='mean')

reg_criterion_none = nn.SmoothL1Loss(reduction='none')
mse_criterion_none = nn.MSELoss(reduction='none')
binary_cls_criterion_none = nn.BCEWithLogitsLoss(reduction='none')

emitter_cls_criterion_none = nn.KLDivLoss(reduction='none')
emitter_cls_criterion_mean = nn.KLDivLoss(reduction='batchmean')

emitter_cls_criterion_L2_none = nn.MSELoss(reduction='none')
emitter_cls_criterion_L2_mean = nn.MSELoss(reduction='mean')

def cls_reg_loss(cls_result, cls_gt, reg_result, reg_gt, cls_reg_ratio):
    cls_loss = cls_criterion(cls_result, cls_gt)
    if len(reg_result.size()) == 3:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1, 1).expand(reg_gt.size(0), 1, reg_gt.size(1)))
    else:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1).expand(reg_gt.size(0), 1))
    reg_result = reg_result.squeeze(1)
    reg_loss = reg_criterion(reg_result, reg_gt)
    return cls_loss, cls_reg_ratio * reg_loss

def cls_reg_loss_none(cls_result, cls_gt, reg_result, reg_gt, cls_reg_ratio):
    cls_loss = cls_criterion_none(cls_result, cls_gt)
    if len(reg_result.size()) == 3:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1, 1).expand(reg_gt.size(0), 1, reg_gt.size(1)))
    else:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1).expand(reg_gt.size(0), 1))
    reg_result = reg_result.squeeze(1)
    reg_loss = reg_criterion_none(reg_result, reg_gt)
    return cls_loss, cls_reg_ratio * reg_loss


def get_labels_dict_layout_emitter(data_batch, opt):
    labels_dict = {}

    if_layout = 'lo' in opt.cfg.DATA.data_read_list
    if_emitter = 'em' in opt.cfg.DATA.data_read_list
    emitter_est_type = opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type

    if if_layout:
        pitch_reg = data_batch['camera']['pitch_reg'].float().cuda(non_blocking=True)
        pitch_cls = data_batch['camera']['pitch_cls'].long().cuda(non_blocking=True)
        roll_reg = data_batch['camera']['roll_reg'].float().cuda(non_blocking=True)
        roll_cls = data_batch['camera']['roll_cls'].long().cuda(non_blocking=True)
        lo_ori_reg = data_batch['layout_reindexed']['ori_reg'].float().cuda(non_blocking=True)
        lo_ori_cls = data_batch['layout_reindexed']['ori_cls'].long().cuda(non_blocking=True)
        lo_centroid = data_batch['layout_reindexed']['centroid_reg'].float().cuda(non_blocking=True)
        lo_coeffs = data_batch['layout_reindexed']['coeffs_reg'].float().cuda(non_blocking=True)
        lo_bdb3D = data_batch['layout_reindexed']['bdb3D'].float().cuda(non_blocking=True)
        cam_K = data_batch['camera']['K'].float().cuda(non_blocking=True)
        cam_R_gt = get_rotation_matrix_gt(opt.bins_tensor, 
                                            pitch_cls, pitch_reg, 
                                            roll_cls, roll_reg)

        layout_labels = {'pitch_reg': pitch_reg, 'pitch_cls': pitch_cls, 'roll_reg': roll_reg,
                        'roll_cls': roll_cls, 'lo_ori_reg': lo_ori_reg, 'lo_ori_cls': lo_ori_cls, 'lo_centroid': lo_centroid,
                        'lo_coeffs': lo_coeffs, 'lo_bdb3D': lo_bdb3D, 'cam_K': cam_K, 'cam_R_gt':cam_R_gt}
        labels_dict['layout_labels'] = layout_labels

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
            cell_light_ratio = data_batch['cell_light_ratio'].cuda(non_blocking=True)
            emitter_labels['cell_light_ratio'] = cell_light_ratio
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

def vis_layout_emitter(labels_dict, output_dict, opt, time_meters):
    output_vis_dict = {}
    if_est_emitter = 'em' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list
    if_est_layout = 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list

    
    batch_size = labels_dict['imBatch'].shape[0]
    grid_size = opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size

    gt_dict_lo = labels_dict['layout_labels']
    pred_dict_lo = output_dict['layout_est_result']
    
    if if_est_layout:
        lo_bdb3D_out = get_layout_bdb_sunrgbd(opt.bins_tensor, pred_dict_lo['lo_ori_reg_result'],
                                            torch.argmax(pred_dict_lo['lo_ori_cls_result'], 1),
                                            pred_dict_lo['lo_centroid_result'],
                                            pred_dict_lo['lo_coeffs_result'])
        cam_R_out = get_rotation_matix_result(opt.bins_tensor,
                                    torch.argmax(pred_dict_lo['pitch_cls_result'], 1), pred_dict_lo['pitch_reg_result'],
                                    torch.argmax(pred_dict_lo['roll_cls_result'], 1), pred_dict_lo['roll_reg_result'])

    if if_est_emitter:
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


    scene_box_list = []
    layout_info_dict_list = []
    emitter_info_dict_list = []
    for sample_idx in range(batch_size):
        # print('--- Visualizing sample %d ---'%sample_idx)
        # save_prefix = 'sample%d-LABEL-epoch%d-tid%d-%s'%(sample_idx+batch_size*vis_batch_count, epoch, iter, phase)
        gt_cam_R = gt_dict_lo['cam_R_gt'][sample_idx].cpu().numpy()
        cam_K = gt_dict_lo['cam_K'][sample_idx].cpu().numpy()
        gt_layout = gt_dict_lo['lo_bdb3D'][sample_idx].cpu().numpy()
        
        if if_est_layout:
            layout_dict = {'layout': lo_bdb3D_out[sample_idx, :, :].cpu().detach().numpy()}
            cam_R_dict = {'cam_R': cam_R_out[sample_idx, :, :].cpu().detach().numpy()}

            pre_layout = layout_dict['layout']                
            pre_cam_R = cam_R_dict['cam_R']
            pre_layout_reindexed = reindex_layout(pre_layout, pre_cam_R)
        else:
            pre_cam_R = gt_cam_R
            pre_layout = gt_layout
            pre_layout_reindexed = gt_layout

        image = (labels_dict['im_trainval_RGB'][sample_idx].detach().cpu().numpy() * 255.).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))

        if if_est_emitter:
            emitter2wall_assign_info_list = gt_dict_em['emitter2wall_assign_info_list'][sample_idx]
            emitters_obj_list = gt_dict_em['emitters_obj_list'][sample_idx]
            gt_layout_RAW = gt_dict_em['gt_layout_RAW'][sample_idx]
            emitter_cls_result_postprocessed_np = emitter_cls_result_postprocessed[sample_idx].detach().cpu().numpy() # [102]
            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
                emitter_cls_prob_GT_np = gt_dict_em['cell_light_ratio'][sample_idx].detach().cpu().numpy()
            else:
                emitter_cls_prob_GT_np = gt_dict_em['emitter_cls_prob'][sample_idx].detach().cpu().numpy()

            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
                face_v_idxes = [(1, 0, 2, 3), (4, 5, 7, 6), (0, 1, 4, 5), (1, 2, 5, 6), (3, 7, 2, 6), (4, 7, 0, 3)]

                cell_info_grid_GT_includeempty = gt_dict_em['cell_info_grid'][sample_idx]
                cell_info_grid_GT = []
                for wall_idx in range(6):
                    for i in range(grid_size):
                        for j in range(grid_size):
                            cell_info = cell_info_grid_GT_includeempty[wall_idx * grid_size**2 + i * grid_size + j]
                            if cell_info['obj_type'] is not None:
                                cell_info['wallidx_i_j'] = (wall_idx, i, j)
                                cell_info_grid_GT.append(cell_info)

                map_obj_type_int = {1: 'window', 2: 'obj', 0: 'null'}
                cell_info_grid_PRED = []
                for wall_idx in range(6):
                    basis_index = face_v_idxes[wall_idx]
                    normal_outside = - np.cross(gt_layout_RAW[basis_index[1]] - gt_layout_RAW[basis_index[0]], gt_layout_RAW[basis_index[2]] - gt_layout_RAW[basis_index[0]]).flatten() # [TODO] change to use EST layout when joint training!!!
                    normal_outside = normal_outside / np.linalg.norm(normal_outside)
                    for i in range(grid_size):
                        for j in range(grid_size):
                            if not opt.cfg.MODEL_LAYOUT_EMITTER.emitter.cls_agnostric:
                                if cell_cls[sample_idx][wall_idx][i][j] == 0:
                                    continue
                            cell_info = {'obj_type': map_obj_type_int[cell_cls[sample_idx][wall_idx][i][j]], 'emitter_info': {}, 'wallidx_i_j': (wall_idx, i, j)}
                            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.relative_dir:
                                cell_info['emitter_info']['light_dir_abs'] = cell_axis[sample_idx][wall_idx][i][j].flatten() + normal_outside
                            else:
                                cell_info['emitter_info']['light_dir_abs'] = cell_axis[sample_idx][wall_idx][i][j].flatten()

                            cell_info['emitter_info']['light_dir_abs'] = cell_info['emitter_info']['light_dir_abs'] / (np.linalg.norm(cell_info['emitter_info']['light_dir_abs'])+1e-6)
                            
                            intensity_log = cell_intensity[sample_idx][wall_idx][i][j].flatten() # actually predicts LOG intensity!
                            assert intensity_log.shape == (3,)
                            intensity = np.exp(intensity_log) - 1.
                            intensity_scale255 = np.amax(intensity) / 255.
                            intensity_scaled255 = intensity / (intensity_scale255 + 1e-5)
                            intensity_scaled01 = [np.clip(x/255., 0., 1.) for x in intensity_scaled255] # max: 1.
                            intensity_scalelog = np.log(np.clip(np.linalg.norm(intensity.flatten()) + 1., 1., np.inf))
                            cell_info['emitter_info']['intensity_scalelog'] = intensity_scalelog # log of norm of intensity
                            cell_info['emitter_info']['intensity_scaled01'] = intensity_scaled01
                            cell_info['emitter_info']['intensity'] = intensity  
                            cell_info['emitter_info']['lamb'] = cell_lamb[sample_idx][wall_idx][i][j].item()
                            cell_info['light_ratio'] = emitter_cls_result_postprocessed_np[wall_idx][i * grid_size + j]
                            cell_info_grid_PRED.append(cell_info)
            else:
                cell_info_grid_PRED, cell_info_grid_GT = None, None
        else:
            emitter2wall_assign_info_list = None
            emitters_obj_list = None
            # gt_layout_RAW = None
            emitter_cls_result_postprocessed_np = None
            emitter_cls_prob_GT_np = None
            cell_info_grid_GT = None
            cell_info_grid_PRED = None

        gt_boxes, pre_boxes = None, None
        save_prefix = ''

        scene_box = Box(image, None, cam_K, gt_cam_R, pre_cam_R, gt_layout, pre_layout_reindexed, gt_boxes, pre_boxes, 'prediction', output_mesh = None, \
            opt=opt, dataset='OR', description=save_prefix, if_mute_print=True, OR=opt.cfg.MODEL_LAYOUT_EMITTER.data.OR, \
            emitter2wall_assign_info_list = emitter2wall_assign_info_list, 
            emitters_obj_list = emitters_obj_list, 
            # gt_layout_RAW = gt_layout_RAW, 
            emitter_cls_prob_PRED = emitter_cls_result_postprocessed_np, 
            emitter_cls_prob_GT = emitter_cls_prob_GT_np, 
            cell_info_grid_GT = cell_info_grid_GT, 
            cell_info_grid_PRED = cell_info_grid_PRED, 
            grid_size = grid_size)

        scene_box_list.append(scene_box)
        layout_info_dict_list.append({'est_data': None, 'gt_cam_R': gt_cam_R, 'cam_K': cam_K, 'gt_layout': gt_layout, 'pre_layout': pre_layout, 'pre_cam_R': pre_cam_R, 'image': image})
        emitter_info_dict = {'cell_info_grid_GT': cell_info_grid_GT, 'cell_info_grid_PRED': cell_info_grid_PRED, \
                                'emitter_cls_prob_PRED': emitter_cls_result_postprocessed_np, 'emitter_cls_prob_GT': emitter_cls_prob_GT_np, \
                                'pre_layout': pre_layout, 'pre_cam_R': pre_cam_R}
        if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.enable:
            envmap_lightAccu_mean = pred_dict_em['envmap_lightAccu_mean'][sample_idx].detach().cpu().numpy()
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


def postprocess_layout_emitter(labels_dict, output_dict, loss_dict, opt, time_meters, if_vis=False):
    if 'em' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
        output_dict, loss_dict = postprocess_emitter(labels_dict, output_dict, loss_dict, opt, time_meters)
    if 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
        output_dict, loss_dict = postprocess_layout(labels_dict, output_dict, loss_dict, opt, time_meters)

    # if if_vis:
    #     output_vis_dict = vis_layout_emitter(labels_dict, output_dict, opt, time_meters)
    #     output_dict['output_vis_dict'] = output_vis_dict

    return output_dict, loss_dict

def postprocess_layout(labels_dict, output_dict, loss_dict, opt, time_meters):
    cls_reg_ratio = opt.cfg.MODEL_LAYOUT_EMITTER.layout.loss.cls_reg_ratio
    pred_dict = output_dict['layout_est_result']
    gt_dict = labels_dict['layout_labels']

    pitch_cls_loss, pitch_reg_loss = cls_reg_loss(pred_dict['pitch_cls_result'], gt_dict['pitch_cls'], \
        pred_dict['pitch_reg_result'], gt_dict['pitch_reg'], cls_reg_ratio)
    roll_cls_loss, roll_reg_loss = cls_reg_loss(pred_dict['roll_cls_result'], gt_dict['roll_cls'], \
        pred_dict['roll_reg_result'], gt_dict['roll_reg'], cls_reg_ratio)
    lo_ori_cls_loss, lo_ori_reg_loss = cls_reg_loss(pred_dict['lo_ori_cls_result'], gt_dict['lo_ori_cls'], \
        pred_dict['lo_ori_reg_result'], gt_dict['lo_ori_reg'], cls_reg_ratio)
    lo_centroid_loss = reg_criterion(pred_dict['lo_centroid_result'], gt_dict['lo_centroid']) * cls_reg_ratio
    lo_coeffs_loss = reg_criterion(pred_dict['lo_coeffs_result'], gt_dict['lo_coeffs']) * cls_reg_ratio

    lo_bdb3D_result = get_layout_bdb_sunrgbd(opt.bins_tensor, pred_dict['lo_ori_reg_result'], gt_dict['lo_ori_cls'], pred_dict['lo_centroid_result'],
                                                pred_dict['lo_coeffs_result'])
    # layout bounding box corner loss
    lo_corner_loss = cls_reg_ratio * reg_criterion(lo_bdb3D_result, gt_dict['lo_bdb3D'])

    layout_losses_dict = {'loss_layout-pitch_cls':pitch_cls_loss, 'loss_layout-pitch_reg':pitch_reg_loss,
            'loss_layout-roll_cls':roll_cls_loss, 'loss_layout-roll_reg':roll_reg_loss,
            'loss_layout-lo_ori_cls':lo_ori_cls_loss, 'loss_layout-lo_ori_reg':lo_ori_reg_loss,
            'loss_layout-lo_centroid':lo_centroid_loss, 'loss_layout-lo_coeffs':lo_coeffs_loss,
            'loss_layout-lo_corner':lo_corner_loss}

    loss_dict.update(layout_losses_dict)
    loss_dict.update({'loss_layout-ALL': sum([layout_losses_dict[x] for x in layout_losses_dict])})

    output_dict.update({'results_layout': {'lo_bdb3D_result': lo_bdb3D_result}})


    return output_dict, loss_dict

def postprocess_emitter(labels_dict, output_dict, loss_dict, opt, time_meters):

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
        emitter_light_ratio_loss = emitter_cls_criterion_L2_mean(emitter_light_ratio_est, emitter_light_ratio_gt)
    else:
        raise ValueError('Unrecognized emitter cls loss type: ' + loss_type)
    
    emitter_light_ratio_loss = emitter_light_ratio_loss * opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss.weight_light_ratio
    emitter_losses_dict, emitter_results_dict = {'loss_emitter-light_ratio': emitter_light_ratio_loss}, {'emitter_light_ratio_est_from_loss': emitter_light_ratio_est}

    if est_type == 'cell_info':
        valid_mask = (labels_dict['emitter_labels']['cell_cls'] != 0).float().view((B, 6, -1))
        window_mask = (labels_dict['emitter_labels']['cell_cls'] == 1).float().view((B, 6, -1))

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
                elif head_name == 'cell_lamb': # lamps have no lambda
                    loss = torch.sum(loss * window_mask) / (torch.sum(window_mask) + 1e-5)
                    loss = loss * opt.cfg.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_lamb


            emitter_losses_dict.update({'loss_emitter-'+head_name: loss})
            emitter_results_dict.update({head_name+'_result_from_loss': emitter_fc_output})

    loss_dict.update(emitter_losses_dict)
    loss_dict.update({'loss_emitter-ALL': sum([emitter_losses_dict[x] for x in emitter_losses_dict])})
    output_dict.update({'results_emitter': emitter_results_dict})
    return output_dict, loss_dict