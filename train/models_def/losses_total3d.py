import torch
import torch.nn as nn
from utils.utils_total3D.utils_OR_layout import get_layout_bdb_sunrgbd
from utils.utils_total3D.utils_OR_cam import get_rotation_matix_result
from utils.utils_total3D.net_utils_libs import get_layout_bdb_sunrgbd, get_bdb_form_from_corners, \
    recover_points_to_world_sys, get_rotation_matix_result, get_bdb_3d_result, \
    get_bdb_2d_result, physical_violation
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance

dist_chamfer = ChamferDistance()

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
def DetLoss(pred_dict, gt_dict, cls_reg_ratio, flattened_valid_mask_tensor=None):
    # calculate loss
    if flattened_valid_mask_tensor is None:
        flattened_valid_mask = [item for sublist in gt_dict['boxes_valid_list'] for item in sublist]
        flattened_valid_mask_tensor = torch.tensor(flattened_valid_mask).float().cuda()
    num_valid_objs = torch.sum(flattened_valid_mask_tensor)

    # if self.config['debug']['debug_loss']:
    #     print('>>> DetLoss', pred_dict['size_reg_result'].shape[0], torch.sum(flattened_valid_mask_tensor), gt_dict['boxes_valid_list'])

    assert pred_dict['size_reg_result'].shape[0] == flattened_valid_mask_tensor.shape[0]

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
def JointLoss(pred_dict, gt_dict, bins_tensor, lo_bdb3D_result, cls_reg_ratio, flattened_valid_mask_tensor=None):
    if flattened_valid_mask_tensor is None:
        flattened_valid_mask = [item for sublist in gt_dict['boxes_valid_list'] for item in sublist]
        flattened_valid_mask_tensor = torch.tensor(flattened_valid_mask).float().to(pred_dict['size_reg_result'].device)
    num_valid_objs = torch.sum(flattened_valid_mask_tensor)
    assert pred_dict['size_reg_result'].shape[0] == flattened_valid_mask_tensor.shape[0]

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

# class ReconLoss(BaseLoss):
    # def __call__(self, est_data, gt_data, extra_results):
def ReconLoss(est_data, gt_data, extra_results, flattened_valid_mask_tensor=None):
    if gt_data['mask_flag'] == 0:
        point_loss = 0.
    else:
        # get the world coordinates for each 3d object.
        bdb3D_form = get_bdb_form_from_corners(extra_results['bdb3D_result'], gt_data['mask_status'])
        obj_points_in_world_sys = recover_points_to_world_sys(bdb3D_form, est_data['meshes'])
        point_loss = 100 * get_point_loss(obj_points_in_world_sys, extra_results['cam_R_result'],
                                            gt_data['K'], gt_data['depth_maps'], bdb3D_form, gt_data['split'],
                                            gt_data['obj_masks'], gt_data['mask_status'])

        # remove samples without depth map
        if torch.isnan(point_loss):
            point_loss = 0.

        mesh_losses_dict = {'loss_mesh-point': point_loss}

    return mesh_losses_dict


# class SVRLoss(BaseLoss):
# @LOSSES.register_module
def SVRLoss_ori(est_dict, gt_dict, subnetworks, face_sampling_rate, flattened_valid_mask_tensor=None):
    if flattened_valid_mask_tensor is None:
        flattened_valid_mask = [item for sublist in gt_dict['boxes_valid_list'] for item in sublist]
        flattened_valid_mask_tensor = torch.tensor(flattened_valid_mask).float().cuda()

    # device = est_dict['mesh_coordinates_results'][0].device
    # chamfer losses
    chamfer_loss = torch.tensor(0.).cuda()
    edge_loss = torch.tensor(0.).cuda()
    boundary_loss = torch.tensor(0.).cuda()
    # print(est_dict.keys())
    # print(flattened_valid_mask_tensor, flattened_valid_mask_tensor.shape)
    m = flattened_valid_mask_tensor

    for stage_id, mesh_coordinates_result in enumerate(est_dict['mesh_coordinates_results']):
        mesh_coordinates_result = mesh_coordinates_result.transpose(1, 2)
        # points to points chamfer loss
        dist1, dist2 = dist_chamfer(gt_dict['mesh_points'], mesh_coordinates_result)[:2]
        # print('---', dist1.shape, dist2.shape)
        chamfer_loss += (torch.mean(dist1)) + (torch.mean(dist2))

        # boundary loss
        if stage_id == subnetworks - 1:
            # print('------>', est_dict['boundary_point_ids'])
            if 1 in est_dict['boundary_point_ids']:
                boundary_loss = torch.mean(dist2[est_dict['boundary_point_ids']])
                # print('------>>', dist2[est_dict['boundary_point_ids']].shape)

        # edge loss
        edge_vec = torch.gather(mesh_coordinates_result, 1,
                                (est_dict['output_edges'][:, :, 0] - 1).unsqueeze(-1).expand(est_dict['output_edges'].size(0),
                                                                                    est_dict['output_edges'].size(1), 3)) \
                    - torch.gather(mesh_coordinates_result, 1,
                                    (est_dict['output_edges'][:, :, 1] - 1).unsqueeze(-1).expand(est_dict['output_edges'].size(0),
                                                                                    est_dict['output_edges'].size(1), 3))

        edge_vec = edge_vec.view(edge_vec.size(0) * edge_vec.size(1), edge_vec.size(2))
        edge_loss += torch.mean(torch.pow(torch.norm(edge_vec, p=2, dim=1), 2))
        # print('---------', torch.pow(torch.norm(edge_vec, p=2, dim=1), 2).shape)

    chamfer_loss = 100 * chamfer_loss / len(est_dict['mesh_coordinates_results'])
    # print('+++', len(est_dict['mesh_coordinates_results']))
    edge_loss = 100 * edge_loss / len(est_dict['mesh_coordinates_results'])
    # print('++++++', len(est_dict['mesh_coordinates_results']))
    boundary_loss = 100 * boundary_loss

    # face distance losses
    face_loss = torch.tensor(0.).cuda()
    for points_from_edges_by_step, points_indicator_by_step in zip(est_dict['points_from_edges'], est_dict['point_indicators']):
        points_from_edges_by_step = points_from_edges_by_step.transpose(1, 2).contiguous()
        _, dist2_face, _, idx2 = dist_chamfer(gt_dict['mesh_points'], points_from_edges_by_step)
        idx2 = idx2.long()
        dist2_face = dist2_face.view(dist2_face.shape[0], dist2_face.shape[1] // face_sampling_rate,
                                        face_sampling_rate)

        # average distance to nearest face.
        dist2_face = torch.mean(dist2_face, dim=2)
        local_dens = gt_dict['densities'][:, idx2[:]][range(gt_dict['densities'].size(0)), range(gt_dict['densities'].size(0)), :]
        in_mesh = (dist2_face <= local_dens).float()
        face_loss += binary_cls_criterion(points_indicator_by_step, in_mesh)

    if est_dict['points_from_edges']:
        face_loss = face_loss / len(est_dict['points_from_edges'])
        # print('+++++++++', len(est_dict['points_from_edges']))

    # return {'chamfer_loss': chamfer_loss, 'face_loss': 0.01 * face_loss,
    #         'edge_loss': 0.1 * edge_loss, 'boundary_loss': 0.5 * boundary_loss}

    mesh_losses_dict = {
        'loss_mesh-chamfer': chamfer_loss, 
        'loss_mesh-face': 0.01 * face_loss,
        'loss_mesh-edge': 0.1 * edge_loss, 
        'loss_mesh-boundary': 0.5 * boundary_loss}

    return mesh_losses_dict


# def SVRLoss_masked(est_dict, gt_dict, subnetworks, face_sampling_rate, flattened_valid_mask_tensor=None):
def SVRLoss(est_dict, gt_dict, subnetworks, face_sampling_rate, flattened_valid_mask_tensor=None):
    if flattened_valid_mask_tensor is None:
        flattened_valid_mask = [item for sublist in gt_dict['boxes_valid_list'] for item in sublist]
        flattened_valid_mask_tensor = torch.tensor(flattened_valid_mask).float().cuda()

    # device = est_dict['mesh_coordinates_results'][0].device
    # chamfer losses
    chamfer_loss = torch.tensor(0.).cuda()
    edge_loss = torch.tensor(0.).cuda()
    boundary_loss = torch.tensor(0.).cuda()
    # print(flattened_valid_mask_tensor, flattened_valid_mask_tensor.shape)
    m_bool = flattened_valid_mask_tensor.bool()
    assert torch.all(m_bool)
    mask = flattened_valid_mask_tensor.unsqueeze(-1) # [N, 1]

    for stage_id, mesh_coordinates_result in enumerate(est_dict['mesh_coordinates_results']):
        mesh_coordinates_result = mesh_coordinates_result.transpose(1, 2)
        # points to points chamfer loss
        dist1, dist2 = dist_chamfer(gt_dict['mesh_points'], mesh_coordinates_result)[:2]
        # print('---', dist1.shape, dist2.shape) # [N, 10000], [N, 2562]
        # chamfer_loss += (torch.mean(dist1)) + (torch.mean(dist2))
        dist1_mean = torch.sum(dist1 * mask.expand_as(dist1)) / (torch.sum(mask.expand_as(dist1))+1e-6)
        dist2_mean = torch.sum(dist2 * mask.expand_as(dist2)) / (torch.sum(mask.expand_as(dist2))+1e-6)
        chamfer_loss += dist1_mean + dist2_mean

        # dist1_RE, dist2_RE = dist_chamfer(gt_dict['mesh_points'][m_bool], mesh_coordinates_result[m_bool])[:2]
        # chamfer_loss_RE = (torch.mean(dist1_RE)) + (torch.mean(dist2_RE))
        # print('<<<<<', dist1_mean + dist2_mean, chamfer_loss_RE)

        # boundary loss
        if stage_id == subnetworks - 1:
            if 1 in est_dict['boundary_point_ids']:
                # assert False
                boundary_loss = torch.mean(dist2[est_dict['boundary_point_ids']])
                print('------>', est_dict['boundary_point_ids'].shape, dist2[est_dict['boundary_point_ids']].shape)
                print('------>>', dist2[est_dict['boundary_point_ids']].shape)

        # edge loss
        # print(mesh_coordinates_result.shape, est_dict['output_edges'].shape)
        # print(mesh_coordinates_result[m_bool].shape, est_dict['output_edges'][m_bool].shape)
        edge_vec = torch.gather(mesh_coordinates_result[m_bool], 1,
                                (est_dict['output_edges'][m_bool][:, :, 0] - 1).unsqueeze(-1).expand(est_dict['output_edges'][m_bool].size(0),
                                                                                    est_dict['output_edges'][m_bool].size(1), 3)) \
                    - torch.gather(mesh_coordinates_result[m_bool], 1,
                                    (est_dict['output_edges'][m_bool][:, :, 1] - 1).unsqueeze(-1).expand(est_dict['output_edges'][m_bool].size(0),
                                                                                    est_dict['output_edges'][m_bool].size(1), 3))

        edge_vec = edge_vec.view(edge_vec.size(0) * edge_vec.size(1), edge_vec.size(2))
        edge_loss += torch.mean(torch.pow(torch.norm(edge_vec, p=2, dim=1), 2))
        # print('---------', torch.pow(torch.norm(edge_vec, p=2, dim=1), 2).shape)

    chamfer_loss = 100 * chamfer_loss / len(est_dict['mesh_coordinates_results'])
    # print('+++', len(est_dict['mesh_coordinates_results']))
    edge_loss = 100 * edge_loss / len(est_dict['mesh_coordinates_results'])
    # print('++++++', len(est_dict['mesh_coordinates_results']))
    boundary_loss = 100 * boundary_loss

    # face distance losses
    face_loss = torch.tensor(0.).cuda()
    for idxx, (points_from_edges_by_step, points_indicator_by_step) in enumerate(zip(est_dict['points_from_edges'], est_dict['point_indicators'])):
        points_from_edges_by_step = points_from_edges_by_step[m_bool]
        points_indicator_by_step = points_indicator_by_step[m_bool]

        points_from_edges_by_step = points_from_edges_by_step.transpose(1, 2).contiguous()
        # print(gt_dict['mesh_points'].shape, gt_dict['densities'][m_bool].shape)
        _, dist2_face, _, idx2 = dist_chamfer(gt_dict['mesh_points'][m_bool], points_from_edges_by_step)
        idx2 = idx2.long()
        dist2_face = dist2_face.view(dist2_face.shape[0], dist2_face.shape[1] // face_sampling_rate,
                                        face_sampling_rate)

        # average distance to nearest face.
        dist2_face = torch.mean(dist2_face, dim=2)
        local_dens = gt_dict['densities'][m_bool][:, idx2[:]][range(gt_dict['densities'][m_bool].size(0)), range(gt_dict['densities'][m_bool].size(0)), :]
        in_mesh = (dist2_face <= local_dens).float()
        # print(points_indicator_by_step.shape, in_mesh.shape, '---')
        face_loss += binary_cls_criterion(points_indicator_by_step, in_mesh)

    if est_dict['points_from_edges']:
        face_loss = face_loss / len(est_dict['points_from_edges'])
        # print('+++++++++', len(est_dict['points_from_edges']))

    # return {'chamfer_loss': chamfer_loss, 'face_loss': 0.01 * face_loss,
    #         'edge_loss': 0.1 * edge_loss, 'boundary_loss': 0.5 * boundary_loss}

    mesh_losses_dict = {
        'loss_mesh-chamfer': chamfer_loss, 
        'loss_mesh-face': 0.01 * face_loss,
        'loss_mesh-edge': 0.1 * edge_loss, 
        'loss_mesh-boundary': 0.5 * boundary_loss}

    return mesh_losses_dict




def get_point_loss(points_in_world_sys, cam_R, cam_K, depth_maps, bdb3D_form, split, obj_masks, mask_status):
    '''
    get the depth loss for each mesh.
    :param points_in_world_sys: Number_of_objects x Number_of_points x 3
    :param cam_R: Number_of_scenes x 3 x 3
    :param cam_K: Number_of_scenes x 3 x 3
    :param depth_maps: Number_of_scenes depth maps in a list
    :param split: Number_of_scenes x 2 matrix
    :return: depth loss
    '''
    depth_loss = 0.
    n_objects = 0
    masked_object_id = -1

    device = points_in_world_sys.device

    for scene_id, obj_interval in enumerate(split):
        # map depth to 3d points in camera system.
        u, v = torch.meshgrid(torch.arange(0, depth_maps[scene_id].size(1)), torch.arange(0, depth_maps[scene_id].size(0)))
        u = u.t().to(device)
        v = v.t().to(device)
        u = u.reshape(-1)
        v = v.reshape(-1)
        z_cam = depth_maps[scene_id][v, u]
        u = u.float()
        v = v.float()

        # non_zero_indices = torch.nonzero(z_cam).t()[0]
        # z_cam = z_cam[non_zero_indices]
        # u = u[non_zero_indices]
        # v = v[non_zero_indices]

        # calculate coordinates
        x_cam = (u - cam_K[scene_id][0][2])*z_cam/cam_K[scene_id][0][0]
        y_cam = (v - cam_K[scene_id][1][2])*z_cam/cam_K[scene_id][1][1]

        # transform to toward-up-right coordinate system
        points_world = torch.cat([z_cam.unsqueeze(-1), -y_cam.unsqueeze(-1), x_cam.unsqueeze(-1)], -1)
        # transform from camera system to world system
        points_world = torch.mm(points_world, cam_R[scene_id].t())

        n_columns = depth_maps[scene_id].size(1)

        for loc_id, obj_id in enumerate(range(*obj_interval)):
            if mask_status[obj_id] == 0:
                continue
            masked_object_id += 1

            bdb2d = obj_masks[scene_id][loc_id]['msk_bdb']
            obj_msk = obj_masks[scene_id][loc_id]['msk']

            u_s, v_s = torch.meshgrid(torch.arange(bdb2d[0], bdb2d[2] + 1), torch.arange(bdb2d[1], bdb2d[3] + 1))
            u_s = u_s.t().long()
            v_s = v_s.t().long()
            index_dep = u_s + n_columns * v_s
            index_dep = index_dep.reshape(-1)
            in_object_indices = obj_msk.reshape(-1).nonzero()[0]

            # remove holes in depth maps
            if len(in_object_indices) == 0:
                continue

            object_pnts = points_world[index_dep,:][in_object_indices,:]
            # remove noisy points that out of bounding boxes
            inner_idx = torch.sum(torch.abs(
                torch.mm(object_pnts - bdb3D_form['centroid'][masked_object_id].view(1, 3), bdb3D_form['basis'][masked_object_id].t())) >
                                  bdb3D_form['coeffs'][masked_object_id], dim=1)

            inner_idx = torch.nonzero(inner_idx == 0).t()[0]

            if inner_idx.nelement() == 0:
                continue

            object_pnts = object_pnts[inner_idx, :]

            dist_1 = dist_chamfer(object_pnts.unsqueeze(0), points_in_world_sys[masked_object_id].unsqueeze(0))[0]
            depth_loss += torch.mean(dist_1)
            n_objects += 1
    return depth_loss/n_objects if n_objects > 0 else torch.tensor(0.).to(device)

