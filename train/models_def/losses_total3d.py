import torch
import torch.nn as nn
from utils.utils_total3D.utils_OR_layout import get_layout_bdb_sunrgbd
from utils.utils_total3D.utils_OR_cam import get_rotation_matix_result
from utils.utils_total3D.net_utils_libs import get_layout_bdb_sunrgbd, get_bdb_form_from_corners, \
    recover_points_to_world_sys, get_rotation_matix_result, get_bdb_3d_result, \
    get_bdb_2d_result, physical_violation

from icecream import ic

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

# class BaseLoss(object):
#     '''base loss class'''
#     def __init__(self, weight=1, config=None):
#         '''initialize loss module'''
#         self.weight = weight
#         self.config = config

# class PoseLoss(BaseLoss):
#     def __call__(self, pred_dict, gt_dict, bins_tensor, cls_reg_ratio):
def PoseLoss(pred_dict, gt_dict, bins_tensor, cls_reg_ratio):
    pitch_cls_loss, pitch_reg_loss = cls_reg_loss(pred_dict['pitch_cls_result'], gt_dict['pitch_cls'], \
        pred_dict['pitch_reg_result'], gt_dict['pitch_reg'], cls_reg_ratio)
    roll_cls_loss, roll_reg_loss = cls_reg_loss(pred_dict['roll_cls_result'], gt_dict['roll_cls'], \
        pred_dict['roll_reg_result'], gt_dict['roll_reg'], cls_reg_ratio)
    lo_ori_cls_loss, lo_ori_reg_loss = cls_reg_loss(pred_dict['lo_ori_cls_result'], gt_dict['lo_ori_cls'], \
        pred_dict['lo_ori_reg_result'], gt_dict['lo_ori_reg'], cls_reg_ratio)
    lo_centroid_loss = reg_criterion(pred_dict['lo_centroid_result'], gt_dict['lo_centroid']) * cls_reg_ratio
    lo_coeffs_loss = reg_criterion(pred_dict['lo_coeffs_result'], gt_dict['lo_coeffs']) * cls_reg_ratio

    lo_bdb3D_result = get_layout_bdb_sunrgbd(bins_tensor, pred_dict['lo_ori_reg_result'], gt_dict['lo_ori_cls'], pred_dict['lo_centroid_result'],
                                                pred_dict['lo_coeffs_result'])
    # layout bounding box corner loss
    lo_corner_loss = cls_reg_ratio * reg_criterion(lo_bdb3D_result, gt_dict['lo_bdb3D'])

    layout_losses_dict = {'loss_layout-pitch_cls':pitch_cls_loss, 'loss_layout-pitch_reg':pitch_reg_loss,
            'loss_layout-roll_cls':roll_cls_loss, 'loss_layout-roll_reg':roll_reg_loss,
            'loss_layout-lo_ori_cls':lo_ori_cls_loss, 'loss_layout-lo_ori_reg':lo_ori_reg_loss,
            'loss_layout-lo_centroid':lo_centroid_loss, 'loss_layout-lo_coeffs':lo_coeffs_loss,
            'loss_layout-lo_corner':lo_corner_loss}

    return layout_losses_dict, lo_bdb3D_result


# class DetLoss(BaseLoss):
    # def __call__(self, pred_dict, gt_dict, cls_reg_ratio):
def DetLoss(pred_dict, gt_dict, cls_reg_ratio):
    # calculate loss
    flattened_valid_mask = [item for sublist in gt_dict['boxes_valid_list'] for item in sublist]
    flattened_valid_mask_tensor = torch.tensor(flattened_valid_mask).float().to(pred_dict['size_reg_result'].device)
    num_valid_objs = torch.sum(flattened_valid_mask_tensor)
    
    # if self.config['debug']['debug_loss']:
    #     print('>>> DetLoss', pred_dict['size_reg_result'].shape[0], torch.sum(flattened_valid_mask_tensor), gt_dict['boxes_valid_list'])

    assert pred_dict['size_reg_result'].shape[0] == len(flattened_valid_mask)

    size_reg_loss = reg_criterion_none(pred_dict['size_reg_result'], gt_dict['size_reg']) * cls_reg_ratio
    ori_cls_loss, ori_reg_loss = cls_reg_loss_none(pred_dict['ori_cls_result'], gt_dict['ori_cls'], pred_dict['ori_reg_result'], gt_dict['ori_reg'], cls_reg_ratio)
    centroid_cls_loss, centroid_reg_loss = cls_reg_loss_none(pred_dict['centroid_cls_result'], gt_dict['centroid_cls'],
                                                        pred_dict['centroid_reg_result'], gt_dict['centroid_reg'], cls_reg_ratio)
    offset_2D_loss = reg_criterion_none(pred_dict['offset_2D_result'], gt_dict['offset_2D'])
    # ic(pred_dict['offset_2D_result'].cpu().detach().numpy()[:10], gt_dict['offset_2D'].cpu().detach().numpy()[:10])
    # print(size_reg_loss.shape, ori_cls_loss.shape, ori_reg_loss.shape, centroid_cls_loss.shape, centroid_reg_loss.shape, offset_2D_loss.shape) # torch.Size([10, 3]) torch.Size([10]) torch.Size([10]) torch.Size([10]) torch.Size([10]) torch.Size([10, 2])
    # print(size_reg_loss.mean(-1).sum(-1).shape, ori_cls_loss.sum(-1).shape, ori_reg_loss.sum(-1).shape, centroid_cls_loss.sum(-1).shape, centroid_reg_loss.sum(-1).shape, offset_2D_loss.mean(-1).sum(-1).shape) # torch.Size([10, 3]) torch.Size([10]) torch.Size([10]) torch.Size([10]) torch.Size([10]) torch.Size([10, 2])
    object_losses_dict = {'loss_object-size_reg':(size_reg_loss.mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs, 
            'loss_object-ori_cls':(ori_cls_loss * flattened_valid_mask_tensor).sum(0)/num_valid_objs, 
            'loss_object-ori_reg':(ori_reg_loss * flattened_valid_mask_tensor).sum(0)/num_valid_objs,
            'loss_object-centroid_cls':(centroid_cls_loss * flattened_valid_mask_tensor).sum(0)/num_valid_objs, 
            'loss_object-centroid_reg':(centroid_reg_loss * flattened_valid_mask_tensor).sum(0)/num_valid_objs,
            'loss_object-offset_2D':(offset_2D_loss.mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs}

    return object_losses_dict

# class JointLoss(BaseLoss):
#     def __call__(self, pred_dict, gt_dict, bins_tensor, layout_results):
def JointLoss(pred_dict, gt_dict, bins_tensor, lo_bdb3D_result, cls_reg_ratio):

    flattened_valid_mask = [item for sublist in gt_dict['boxes_valid_list'] for item in sublist]
    flattened_valid_mask_tensor = torch.tensor(flattened_valid_mask).float().to(pred_dict['size_reg_result'].device)
    num_valid_objs = torch.sum(flattened_valid_mask_tensor)
    assert pred_dict['size_reg_result'].shape[0] == len(flattened_valid_mask)

    # print(pred_dict['size_reg_result'].shape, torch.sum(flattened_valid_mask_tensor), gt_dict['boxes_valid_list'])

    # predicted camera rotation
    cam_R_result = get_rotation_matix_result(bins_tensor,
                                                gt_dict['pitch_cls'], pred_dict['pitch_reg_result'],
                                                gt_dict['roll_cls'], pred_dict['roll_reg_result'])

    # projected center
    P_result = torch.stack(
        ((gt_dict['bdb2D_pos'][:, 0] + gt_dict['bdb2D_pos'][:, 2]) / 2 - (gt_dict['bdb2D_pos'][:, 2] - gt_dict['bdb2D_pos'][:, 0]) * pred_dict['offset_2D_result'][:, 0],
            (gt_dict['bdb2D_pos'][:, 1] + gt_dict['bdb2D_pos'][:, 3]) / 2 - (gt_dict['bdb2D_pos'][:, 3] - gt_dict['bdb2D_pos'][:, 1]) * pred_dict['offset_2D_result'][:, 1]), 1)

    # retrieved 3D bounding box
    bdb3D_result, _ = get_bdb_3d_result(bins_tensor,
                                        gt_dict['ori_cls'],
                                        pred_dict['ori_reg_result'],
                                        gt_dict['centroid_cls'],
                                        pred_dict['centroid_reg_result'],
                                        gt_dict['size_cls'],
                                        pred_dict['size_reg_result'],
                                        P_result,
                                        gt_dict['cam_K_scaled'],
                                        cam_R_result,
                                        gt_dict['split'])


    # 3D bounding box corner loss
    corner_loss = 5 * cls_reg_ratio * reg_criterion_none(bdb3D_result, gt_dict['bdb3D'])

    # 2D bdb loss
    bdb2D_result = get_bdb_2d_result(bdb3D_result, cam_R_result, gt_dict['cam_K_scaled'], gt_dict['split'])
    bdb2D_loss = 20 * cls_reg_ratio * reg_criterion_none(bdb2D_result, gt_dict['bdb2D_from_3D_gt'])

    # physical violation loss
    phy_violation, phy_gt = physical_violation(lo_bdb3D_result, bdb3D_result, gt_dict['split'])
    phy_loss = 20 * mse_criterion_none(phy_violation, phy_gt)

    # print(((corner_loss.mean(-1).mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs).shape, \
    #     ((bdb2D_loss.mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs).shape, \
    #     ((phy_loss.mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs).shape) # torch.Size([10, 8, 3]) torch.Size([10, 4]) torch.Size([10, 3])
    # print(((corner_loss.mean(-1).mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs), \
    #     ((bdb2D_loss.mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs), \
    #     ((phy_loss.mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs)) # torch.Size([10, 8, 3]) torch.Size([10, 4]) torch.Size([10, 3])
    joint_losses_dict = {
        'loss_joint-phy': (phy_loss.mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs, \
        'loss_joint-bdb2D': (bdb2D_loss.mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs, \
        'loss_joint-corner': (corner_loss.mean(-1).mean(-1) * flattened_valid_mask_tensor).sum(0)/num_valid_objs
        }
    joint_outputs_dict = {'cam_R_result': cam_R_result, 'bdb3D_result': bdb3D_result}
    # print(return_dict)

    return joint_losses_dict, joint_outputs_dict
