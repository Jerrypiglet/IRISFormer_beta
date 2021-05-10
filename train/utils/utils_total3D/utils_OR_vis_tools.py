import sys

from numpy.core.numeric import True_
sys.path.append('.')
import argparse
# from utils.sunrgbd_config import SUNRGBD_CONFIG
import os
import json
import pickle
from utils.utils_total3D.data_config import Dataset_Config, NYU40CLASSES, OR4XCLASSES_dict, RECON_3D_CLS, RECON_3D_CLS_OR_dict

import numpy as np
# from libs.tools import R_from_yaw_pitch_roll
import scipy.io as sio
from glob import glob
os.environ['LD_LIBRARY_PATH'] = '/home/ruizhu/anaconda3/envs/semanticInverse/lib'
# from utils.vis_tools import Scene3D, nyu_color_palette
from utils.utils_total3D.vis_tools import Scene3D, nyu_color_palette
from utils.utils_total3D.sunrgbd_utils import proj_from_point_to_2d, get_corners_of_bb3d_no_index

from PIL import Image, ImageDraw, ImageFont
from utils.utils_total3D.utils_OR_cam import project_3d_line
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from icecream import ic

import matplotlib.patches as patches
from utils.utils_total3D.utils_OR_geo import bb_intersection_over_union

from utils.utils_misc import *
from utils.utils_total3D.utils_rui import vis_axis_xyz, Arrow3D, vis_axis, vis_cube_plt
from utils.utils_total3D.utils_OR_vis_labels import set_axes_equal
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.utils_misc import yellow, magenta, white_blue

from SimpleLayout.utils_SL import SimpleScene

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utils.utils_total3D.utils_OR_mesh import loadMesh, writeMesh

def get_bdb_form_from_corners(corners):
    vec_0 = (corners[:, 2, :] - corners[:, 1, :]) / 2.
    vec_1 = (corners[:, 0, :] - corners[:, 4, :]) / 2.
    vec_2 = (corners[:, 1, :] - corners[:, 0, :]) / 2.

    coeffs_0 = np.linalg.norm(vec_0, axis=1)
    coeffs_1 = np.linalg.norm(vec_1, axis=1)
    coeffs_2 = np.linalg.norm(vec_2, axis=1)
    coeffs = np.stack([coeffs_0, coeffs_1, coeffs_2], axis=1)

    centroid = (corners[:, 0, :] + corners[:, 6, :]) / 2.

    basis_0 = np.dot(np.diag(1 / coeffs_0), vec_0)
    basis_1 = np.dot(np.diag(1 / coeffs_1), vec_1)
    basis_2 = np.dot(np.diag(1 / coeffs_2), vec_2)

    basis = np.stack([basis_0, basis_1, basis_2], axis=1)

    return {'basis': basis, 'coeffs': coeffs, 'centroid': centroid}


def format_bboxes(box, type):
    # box['bdb']: array, [N, 8, 3]
    # box['size_cls']: list, [N]
    assert isinstance(box['class_id'], list)

    if type == 'prediction':
        boxes = {}
        basis_list = []
        centroid_list = []
        coeff_list = []

        # convert bounding boxes
        box_data = box['bdb']

        for index in range(len(box_data)):
            # print(box_data[index])
            basis = box_data[index]['basis']
            centroid = box_data[index]['centroid']
            coeffs = box_data[index]['coeffs']
            basis_list.append(basis.reshape((3, 3)))
            centroid_list.append(centroid.reshape((3,)))
            coeff_list.append(coeffs.reshape((3, 1)))

        boxes['basis'] = np.stack(basis_list, 0)
        boxes['centroid'] = np.stack(centroid_list, 0)
        boxes['coeffs'] = np.stack(coeff_list, 0)
        boxes['class_id'] = box['class_id']

        assert len(box_data) == len(boxes['class_id'])

    elif type == 'GT':
        assert box['bdb3D'].shape[0] == len(box['class_id'])
        boxes = get_bdb_form_from_corners(box['bdb3D'])
        boxes['class_id'] = box['class_id']

    return boxes

def format_layout(layout_data):

    layout_bdb = {}

    centroid = (layout_data.max(0) + layout_data.min(0)) / 2.

    vector_z = (layout_data[1] - layout_data[0]) / 2.
    coeff_z = np.linalg.norm(vector_z)
    basis_z = vector_z/coeff_z

    vector_x = (layout_data[2] - layout_data[1]) / 2.
    coeff_x = np.linalg.norm(vector_x)
    basis_x = vector_x/coeff_x

    vector_y = (layout_data[0] - layout_data[4]) / 2.
    coeff_y = np.linalg.norm(vector_y)
    basis_y = vector_y/coeff_y

    basis = np.array([basis_x, basis_y, basis_z])
    coeffs = np.array([coeff_x, coeff_y, coeff_z])

    layout_bdb['coeffs'] = coeffs
    layout_bdb['centroid'] = centroid
    layout_bdb['basis'] = basis

    return layout_bdb

def format_mesh(obj_files, bboxes, if_use_vtk=False, validate_classids=False):

    if if_use_vtk:
        vtk_objects = {}
    else:
        vertices_list = []
        faces_list = []
        num_vertices = 0

    for idx, obj_file in enumerate(obj_files):
        # print(obj_file)
        if validate_classids:
            filename = '.'.join(os.path.basename(obj_file).split('.')[:-1])
            obj_idx = int(filename.split('_')[0])
            class_id = int(filename.split('_')[1].split(' ')[0])
            assert bboxes['class_id'][obj_idx] == class_id
        else:
            obj_idx = idx
        
        # print(obj_file)
        if if_use_vtk:
            object = vtk.vtkOBJReader()
            object.SetFileName(obj_file)
            # print(obj_file)
            object.Update()

            # get points from object
            polydata = object.GetOutput()
            # read points using vtk_to_numpy
            points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)
            # print(points.shape)
        else:
            points, faces = loadMesh(obj_file)

        # normalize points
        mesh_center = (points.max(0) + points.min(0)) / 2.
        points = points - mesh_center

        mesh_coef = (points.max(0) - points.min(0)) / 2.
        points = points.dot(np.diag(1./mesh_coef)).dot(np.diag(bboxes['coeffs'][obj_idx]))

        # set orientation
        points = points.dot(bboxes['basis'][obj_idx])

        # move to center
        points = points + bboxes['centroid'][obj_idx]


        if if_use_vtk:
            points_array = numpy_to_vtk(points, deep=True)
            polydata.GetPoints().SetData(points_array)
            object.Update()

            vtk_objects[obj_idx] = object
        else:
            points_swapped = points.copy()
            # points_swapped[:, 2] = -points_swapped[:, 2] # for OR
            vertices_list.append(points_swapped)
            faces_list.append(faces+num_vertices)
            num_vertices += points.shape[0]

    if if_use_vtk:
        return vtk_objects, bboxes
    else:
        return [vertices_list, faces_list], bboxes
