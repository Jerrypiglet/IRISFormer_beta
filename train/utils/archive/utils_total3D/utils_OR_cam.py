import numpy as np
import os.path as osp
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
# print(sys.path)
from utils.utils_total3D.utils_OR_geo import isect_line_plane_v3
import torch

# ======= total3d

def get_rotation_matix_result(bins_tensor, pitch_cls_result, pitch_reg_result, roll_cls_result, roll_reg_result, if_differentiable=False, if_input_after_argmax=False, if_reg_from_gt_cls=False):
    '''
    get rotation matrix from predicted camera pitch, roll angles.
    '''
    if if_input_after_argmax:
        pitch_cls = pitch_cls_result
        roll_cls = roll_cls_result
        assert if_differentiable==False
    else:
        pitch_cls = torch.argmax(pitch_cls_result, 1)
        roll_cls = torch.argmax(roll_cls_result, 1)

    if if_reg_from_gt_cls:
        pitch_result = pitch_reg_result
        roll_result = roll_reg_result
    else:
        pitch_result = torch.gather(pitch_reg_result, 1,
                                pitch_cls.view(pitch_cls.size(0), 1).expand(pitch_cls.size(0), 1)).squeeze(1)
        roll_result = torch.gather(roll_reg_result, 1,
                                roll_cls.view(roll_cls.size(0), 1).expand(roll_cls.size(0), 1)).squeeze(1)

    if if_differentiable:
        pitch = num_from_bins_differentiable(bins_tensor['pitch_bin'], pitch_cls_result, pitch_result)
        roll = num_from_bins_differentiable(bins_tensor['roll_bin'], roll_cls_result, roll_result)
    else:
        pitch = num_from_bins(bins_tensor['pitch_bin'], pitch_cls, pitch_result)
        roll = num_from_bins(bins_tensor['roll_bin'], roll_cls, roll_result)

    cam_R = R_from_yaw_pitch_roll(torch.zeros_like(pitch), pitch, roll)
    
    return cam_R

def get_rotation_matrix_gt(bins_tensor, pitch_cls_gt, pitch_reg_gt, roll_cls_gt, roll_reg_gt):
    '''
    get rotation matrix from predicted camera pitch, roll angles.
    '''
    pitch = num_from_bins(bins_tensor['pitch_bin'], pitch_cls_gt, pitch_reg_gt)
    roll = num_from_bins(bins_tensor['roll_bin'], roll_cls_gt, roll_reg_gt)
    r_ex = R_from_yaw_pitch_roll(torch.zeros_like(pitch), pitch, roll)
    return r_ex

def num_from_bins(bins, cls, reg):
    """
    :param bins: b x 2 tensors
    :param cls: b long tensors
    :param reg: b tensors
    :return: bin_center: b tensors
    """
    bin_width = (bins[0][1] - bins[0][0])
    bin_center = (bins[cls, 0] + bins[cls, 1]) / 2
    return bin_center + reg * bin_width

def num_from_bins_differentiable(bins, cls_results, reg):
    """
    :param bins: batchsize x 2 tensors
    # :param cls: batchsize long tensors
    :param cls_prob: batchsize x nbins(2) tensors
    :param reg: batchsize tensors
    :return: bin_center: batchsize tensors
    """
    bin_width = (bins[0][1] - bins[0][0])
    cls_probs = torch.softmax(cls_results, dim=1) # [batchsize, nbins]
    # print(cls_probs.shape, cls.shape, bins.shape, bins)
    # bin_center = (bins[cls, 0] + bins[cls, 1]) / 2
    bin_centers = (bins[:, 0] + bins[:, 1]) / 2
    bin_centers = bin_centers.view(1, -1) # [1, nbins]
    bin_cls_softargmax = (cls_probs * bin_centers).sum(1) # [batchsize]

    return bin_cls_softargmax + reg * bin_width

def R_from_yaw_pitch_roll(yaw, pitch, roll):
    '''
    get rotation matrix from predicted camera yaw, pitch, roll angles.
    :param yaw: batch_size x 1 tensor
    :param pitch: batch_size x 1 tensor
    :param roll: batch_size x 1 tensor
    :return: camera rotation matrix
    '''
    n = yaw.size(0)
    R = torch.zeros((n, 3, 3), device=yaw.device)
    R[:, 0, 0] = torch.cos(yaw) * torch.cos(pitch)
    R[:, 0, 1] = torch.sin(yaw) * torch.sin(roll) - torch.cos(yaw) * torch.cos(roll) * torch.sin(pitch)
    R[:, 0, 2] = torch.cos(roll) * torch.sin(yaw) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
    R[:, 1, 0] = torch.sin(pitch)
    R[:, 1, 1] = torch.cos(pitch) * torch.cos(roll)
    R[:, 1, 2] = - torch.cos(pitch) * torch.sin(roll)
    R[:, 2, 0] = - torch.cos(pitch) * torch.sin(yaw)
    R[:, 2, 1] = torch.cos(yaw) * torch.sin(roll) + torch.cos(roll) * torch.sin(yaw) * torch.sin(pitch)
    R[:, 2, 2] = torch.cos(yaw) * torch.cos(roll) - torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)

    return R

# ======= Rui

def read_cam_params(camFile):
    assert osp.isfile(str(camFile))
    with open(str(camFile), 'r') as camIn:
    #     camNum = int(camIn.readline().strip() )
        cam_data = camIn.read().splitlines()
    cam_num = int(cam_data[0])
    cam_params = np.array([x.split(' ') for x in cam_data[1:]]).astype(np.float)
    assert cam_params.shape[0] == cam_num * 3
    cam_params = np.split(cam_params, cam_num, axis=0) # [[origin, lookat, up], ...]
    return cam_params

def normalize(x):
    return x / np.linalg.norm(x)

def project_v(v, cam_R, cam_t, cam_K, if_only_proj_front_v=False, if_return_front_flags=False, if_v_already_transformed=False, extra_transform_matrix=np.eye(3)):
    if if_v_already_transformed:
        v_transformed = v.T
    else:
        v_transformed = cam_R @ v.T + cam_t
    
    v_transformed = (v_transformed.T @ extra_transform_matrix).T
#     print(v_transformed[2:3, :])
    if if_only_proj_front_v:
        v_transformed = v_transformed * (v_transformed[2:3, :] > 0.)
    p = cam_K @ v_transformed
    if not if_return_front_flags:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T
    else:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T, (v_transformed[2:3, :] > 0.).flatten().tolist()

def project_3d_line(x1x2, cam_R, cam_t, cam_K, cam_center, cam_zaxis, if_debug=False, extra_transform_matrix=np.eye(3)):
    assert len(x1x2.shape)==2 and x1x2.shape[1]==3
    # print(cam_R.shape, x1x2.T.shape, cam_t.shape)
    x1x2_transformed = (cam_R @ x1x2.T + cam_t).T @ extra_transform_matrix
    # print(x1x2_transformed)
    if if_debug:
        print('x1x2_transformed', x1x2_transformed)
    front_flags = list(x1x2_transformed[:, -1] > 0.)
    if if_debug:
        print('front_flags', front_flags)
    if not all(front_flags):
        if not front_flags[0] and not front_flags[1]:
            return None
        x_isect = isect_line_plane_v3(x1x2[0], x1x2[1], cam_center, cam_zaxis, epsilon=1e-6)
#             print(x1x2[front_flags.index(True)], x_isect)
        x1x2 = np.vstack((x1x2[front_flags.index(True)].reshape((1, 3)), x_isect.reshape((1, 3))))
        x1x2_transformed = (cam_R @ x1x2.T + cam_t).T @ extra_transform_matrix
        # print('-->', x1x2_transformed)
    if if_debug:
        print('x1x2_transformed after', x1x2_transformed)

    # x1x2_transformed = x1x2_transformed @ extra_transform_matrix
    # print(x1x2_transformed)
    p = cam_K @ x1x2_transformed.T
    if not if_debug:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T
    else:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T, x1x2

# def project_v_homo(v, cam_transformation4x4, cam_K):
#     # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/img30.gif
#     # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
#     v_homo = np.hstack([v, np.ones((v.shape[0], 1))])
#     cam_K_homo = np.hstack([cam_K, np.zeros((3, 1))])
# #     v_transformed = cam_R @ v.T + cam_t

#     v_transformed = cam_transformation4x4 @ v_homo.T
#     v_transformed_nonhomo = np.vstack([v_transformed[0, :]/v_transformed[3, :], v_transformed[1, :]/v_transformed[3, :], v_transformed[2, :]/v_transformed[3, :]])
# #     print(v_transformed.shape, v_transformed_nonhomo.shape)
#     v_transformed = v_transformed * (v_transformed_nonhomo[2:3, :] > 0.)
#     p = cam_K_homo @ v_transformed
#     return np.vstack([p[0, :]/p[2, :], p[1, :]/p[2, :]]).T
