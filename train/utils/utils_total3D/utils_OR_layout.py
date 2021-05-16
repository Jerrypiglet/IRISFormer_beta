import torch
import torch.nn as nn
import copy
import numpy as np
from utils.utils_total3D.utils_OR_cam import num_from_bins, num_from_bins_differentiable
from sympy import Point3D, Line3D, Plane, sympify, Rational
from utils.utils_total3D.utils_OR_geo import rotation_matrix_from_vectors

def to_dict_tensor(dicts, if_cuda):
    '''
    Store dict to torch tensor.
    :param dicts:
    :param if_cuda:
    :return:
    '''
    dicts_new = copy.copy(dicts)
    for key, value in dicts_new.items():
        value_new = torch.from_numpy(np.array(value))
        if value_new.type() == 'torch.DoubleTensor':
            value_new = value_new.float()
        if if_cuda:
            value_new = value_new.cuda()
        dicts_new[key] = value_new
    return dicts_new

def layout_basis_from_ori(ori):
    """
    :param ori: orientation angle
    :return: basis: 3x3 matrix
            the basis in 3D coordinates
    """
    n = ori.size(0)

    basis = torch.zeros((n, 3, 3)).cuda()

    basis[:, 0, 0] = torch.sin(ori)
    basis[:, 0, 2] = torch.cos(ori)
    basis[:, 1, 1] = 1
    basis[:, 2, 0] = -torch.cos(ori)
    basis[:, 2, 2] = torch.sin(ori)

    return basis

def get_corners_of_bb3d(basis, coeffs, centroid):
    """
    :param basis: n x 3 x 3 tensor
    :param coeffs: n x 3 tensor
    :param centroid:  n x 3 tensor
    :return: corners n x 8 x 3 tensor
    """
    n = basis.size(0)
    corners = torch.zeros((n, 8, 3)).cuda()
    coeffs = coeffs.view(n, 3, 1).expand(-1, -1, 3)
    centroid = centroid.view(n, 1, 3).expand(-1, 8, -1)
    corners[:, 0, :] = - basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] - basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 1, :] = - basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 2, :] =   basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 3, :] =   basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] - basis[:, 2, :] * coeffs[:, 2, :]

    corners[:, 4, :] = - basis[:, 0, :] * coeffs[:, 0, :] - basis[:, 1, :] * coeffs[:, 1, :] - basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 5, :] = - basis[:, 0, :] * coeffs[:, 0, :] - basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 6, :] =   basis[:, 0, :] * coeffs[:, 0, :] - basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 7, :] =   basis[:, 0, :] * coeffs[:, 0, :] - basis[:, 1, :] * coeffs[:, 1, :] - basis[:, 2, :] * coeffs[:, 2, :]
    corners = corners + centroid

    return corners

def get_layout_bdb_sunrgbd(bins_tensor, lo_ori_reg, lo_ori_cls_result, centroid_reg, coeffs_reg, if_return_full=False, if_differentiable=False, if_input_after_argmax=False):
    """
    get the eight corners of 3D bounding box
    :param bins_tensor:
    :param lo_ori_reg: layout orientation regression results
    :param lo_ori_cls: layout orientation classification results
    :param centroid_reg: layout centroid regression results
    :param coeffs_reg: layout coefficients regression results
    :return: bdb: b x 8 x 3 tensor: the bounding box of layout in layout system.
    """
    if if_input_after_argmax:
        lo_ori_cls = lo_ori_cls_result
        assert if_differentiable==False
    else:
        lo_ori_cls = torch.argmax(lo_ori_cls_result, dim=1)

    ori_reg = torch.gather(lo_ori_reg, 1, lo_ori_cls.view(lo_ori_cls.size(0), 1).expand(lo_ori_cls.size(0), 1)).squeeze(1)
    if if_differentiable:
        ori = num_from_bins_differentiable(bins_tensor['layout_ori_bin'], lo_ori_cls_result, ori_reg)
    else:
        ori = num_from_bins(bins_tensor['layout_ori_bin'], lo_ori_cls, ori_reg)

    basis = layout_basis_from_ori(ori)

    centroid_reg = centroid_reg + bins_tensor['layout_centroid_avg']

    coeffs_reg = (coeffs_reg + 1) * bins_tensor['layout_coeffs_avg']

    bdb = get_corners_of_bb3d(basis, coeffs_reg, centroid_reg)

    if if_return_full:
        return bdb, basis, coeffs_reg, centroid_reg
    else:
        return bdb


def shift_left(seq, n):
    return seq[n:]+seq[:n]

def reindex_layout(layout_bbox_3d, cam_R):
    xaxis, yaxis, zaxis = np.split(cam_R, 3, axis=1)

    p1 = Point3D(0., 0., 0.)
    p2 = Point3D(yaxis[0][0], yaxis[1][0], yaxis[2][0])
    p3 = Point3D(zaxis[0][0], zaxis[1][0], zaxis[2][0])
    plane_c = Plane(p1, p2, p3)
    dists_c_v = []
    for idx_v, v in enumerate(layout_bbox_3d[:4]):
        v_2d_P = Point3D(v[0], 0., v[2])
        xaxis_2d = xaxis[[0, 2], :].flatten()
        v_2d = v.reshape((3, 1))[[0, 2], :].flatten()
        dist = plane_c.distance(v_2d_P).evalf() * np.sign(np.dot(xaxis_2d, v_2d))
        dists_c_v.append(dist)
    left_most_index = dists_c_v.index(max(dists_c_v)) # the new #0 vertex

    new_edge_01_3d = layout_bbox_3d[(left_most_index+1)%4, :] - layout_bbox_3d[left_most_index, :] # [x_1-x_0, 0., z_1-z_0], need to make it align with x axis
    assert np.max(np.abs(new_edge_01_3d[1])) < 1e-4
    x_axis_target = np.array([1., 0., 0.])
    r = rotation_matrix_from_vectors(new_edge_01_3d, x_axis_target)
    trans_3x3 = r

    # layout_bbox_3d_aligned = layout_bbox_3d @ r.T

    # === renidex layout
    new_idx_list1 = shift_left([0, 1, 2, 3], left_most_index)
    new_idx_list2 = shift_left([4, 5, 6, 7], left_most_index)
    layout_bbox_3d_reindexed = np.vstack([layout_bbox_3d[new_idx_list1, :], layout_bbox_3d[new_idx_list2, :]])
    
    return layout_bbox_3d_reindexed
