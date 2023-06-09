import numpy as np
import os
import pickle
from PIL import Image
import json
from scipy.io import loadmat
from utils.utils_total3D.libs.tools import get_world_R, normalize_point, yaw_pitch_roll_from_R, R_from_yaw_pitch_roll
# from utils.sunrgbd_config import SUNRGBD_CONFIG, SUNRGBD_DATA
# import pandas as pd
# import jellyfish as jf
from copy import deepcopy
import cv2
# from utils.utils_rui import PolygonArea

# sunrgbd_config = SUNRGBD_CONFIG()
# class_mapping = pd.read_csv(sunrgbd_config.class_mapping_file).drop(['Unnamed: 0'], axis=1)

def get_cam_KRT(cam_paras, im_size):
    '''
    Get the camera intrinsic matrix, rotation matrix and origin point.

    A point [x, y, z] in world coordinate system can be transformed to the camera system by:
    [x, y, z].dot(R)
    :param cam_paras: camera parameters with SUNCG form: [ori_pnt[0], ori_pnt[1], ori_pnt[2], x[0], x[1], x[2], y[0], y[1], y[2], fov_x, fov_y]
    :param im_size: [width, height] of an image.
    :return: R, ori_pnt
    '''
    ori_pnt = cam_paras[:3]
    toward = cam_paras[3:6]  # x-axis
    toward /= np.linalg.norm(toward)
    up = cam_paras[6:9]  # y-axis
    up /= np.linalg.norm(up)
    right = np.cross(toward, up)  # z-axis
    right /= np.linalg.norm(right)
    R = np.vstack([toward, up, right]).T  # columns respectively corresponds to toward, up, right vectors.

    fov_x = cam_paras[9]
    fov_y = cam_paras[10]
    width = im_size[0]
    height = im_size[1]

    f_x = width / (2 * np.tan(fov_x))
    f_y = height / (2 * np.tan(fov_y))

    K = np.array([[f_x, 0., (width-1)/2.], [0., f_y, (height-1)/2.], [0., 0., 1.]])

    return K, R, ori_pnt

def rotate_towards_cam_front(normal, point, frontal_basis_id):
    '''
    roate normal in horizontal plane with pi/2 to make it the same direction with point.
    '''

    rot_matrix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    rotated_normals = [np.linalg.matrix_power(rot_matrix, i).dot(normal) for i in range(4)]

    max_dot_value = 0.
    best_normal = None
    best_hori_id = None

    hori_id = 1 - frontal_basis_id
    for vector in rotated_normals:
        dot_value = vector.dot(point)
        hori_id = 1-hori_id
        if dot_value > max_dot_value:
            max_dot_value = dot_value
            best_normal = vector
            best_hori_id = hori_id

    return best_normal, best_hori_id

def get_layout_info(layout_3D, cam_front):
    '''
    get the layout bbox center, sizes and orientation.
    We rotate the forward vector of layout (by pi/2), to make its dot product (with camera forward vector) to be maximal.
    '''
    center = layout_3D['centroid']
    vectors = layout_3D['vectors']
    coeffs = np.linalg.norm(vectors,axis=1)
    basis = np.array([vector/np.linalg.norm(vector) for vector in vectors])

    # frontal axis
    horizontal_dims = [0, 2]  # must be two dimensional. It means x and z axis are the horizontal axes.
    horizontal_id = 0         # we rotate the x-axis (horizontal_dims[horizontal_id]) toward cam front.
    frontal_basis = basis[0, : ]
    # print(frontal_basis, cam_front)
    # basis_origin, _ = rotate_towards_cam_front(layout_3D['basis_origin'], cam_front, horizontal_id)
    frontal_basis, horizontal_id = rotate_towards_cam_front(frontal_basis, cam_front, horizontal_id)
    # print(frontal_basis, horizontal_id)

    up_basis = basis[1, : ]
    right_basis = np.cross(frontal_basis, up_basis)

    frontal_coeff = coeffs[horizontal_dims[horizontal_id]]
    up_coeff = coeffs[1]
    right_coeff = coeffs[horizontal_dims[1-horizontal_id]]

    layout = {}
    layout['centroid'] = center
    layout['coeffs'] = np.array([frontal_coeff, up_coeff, right_coeff])
    layout['basis'] = np.vstack([frontal_basis, up_basis, right_basis])
    # layout['basis_origin'] = basis_origin

    return layout


def correct_flipped_objects(obj_points, transform_matrix, model_path, voxels=None, sampled_points=None, flipped_objects_in_sunrgbd=[]):
    '''
    correct those wrongly labeled objects to correct orientation.
    :param obj_points: obj points
    :param model_path: the path of the obj model.
    :param transform_matrix: original transfrom matrix from object system to world system
    :return:
    '''

    # These objects are with an opposite frontal direction.

    if model_path.split('/')[-1] in flipped_objects_in_sunrgbd:

        R = np.array([[-1.,  0.,  0.],
                      [ 0.,  1.,  0.],
                      [ 0.,  0., -1.]])
        obj_points = obj_points.dot(R)

        transform_matrix[:3,:3] = transform_matrix[:3,:3].dot(R)

        if isinstance(voxels, np.ndarray):
            voxels = np.rot90(voxels, 2, (0, 2))
        if isinstance(sampled_points, np.ndarray):
            sampled_points = sampled_points.dot(R)

    return obj_points, transform_matrix, voxels, sampled_points


def proj_from_point_to_2d(_points, _K, _R):
    '''
    To project 3d points from world system to 2D image plane.
    Note: The origin center of world system has been moved to the cam center.
    :param points: Nx3 vector
    :param K: 3x3 intrinsic matrix
    :param R: Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the toward, up,
              right vector relative to the world system.
    :return:
    '''
    points = np.copy(_points)
    K = np.copy(_K)
    R = np.copy(_R)

    D_FLAG = 0
    if len(points.shape) == 1:
        points = points[None, :]
        D_FLAG = 1

    p_cam = points.dot(R)
    # convert to traditional image coordinate system
    T_cam = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
    p_cam = p_cam.dot(T_cam.T)

    # delete those points whose depth value is non-positive.
    invalid_ids = np.where(p_cam[:,2]<=0)[0]
    p_cam[invalid_ids, 2] = 0.0001

    p_cam_h = p_cam/p_cam[:,2][:, None]
    pixels = (K.dot(p_cam_h.T)).T

    if D_FLAG == 1:
        pixels = pixels[0][:2]
    else:
        pixels = pixels[:, :2]

    return pixels, invalid_ids

def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[1, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]

    corners[4, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[5, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[6, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[7, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners


def read_seg2d_data(seg2d_path):

    # load seg 2d data.
    try:
        with open(seg2d_path, encoding='utf-8') as data_file:
            seg2d_data = json.load(data_file)
            # print(seg2d_path)
    except Exception as err:
        print('==== Error with '+seg2d_path, err)
        with open(seg2d_path, 'r') as data_file:
            content = data_file.readlines()[0]
        if "\\" in content:
            error_string = "\\"
        else:
            error_string = content[err.pos - 1:err.pos + 7]
        content = content.replace(error_string, "")
        try:
            seg2d_data = json.loads(content)
        except json.decoder.JSONDecodeError:
            print('Error loading json file %s!'%content)

    number_of_anot = len(seg2d_data["frames"][0]["polygon"])

    seg_list = []

    for i in range(number_of_anot):
        x = seg2d_data["frames"][0]["polygon"][i]["x"]
        y = seg2d_data["frames"][0]["polygon"][i]["y"]
        idx_obj = seg2d_data["frames"][0]["polygon"][i]["object"]

        if idx_obj >= len(seg2d_data['objects']):
            continue

        label = seg2d_data['objects'][idx_obj]["name"].lower()
        label = ''.join(i for i in label if not i.isdigit())

        if type(x) != list or type(y) != list:
            continue

        all_points_x = list(map(round, x))
        all_points_y = list(map(round, y))

        seg_data = {'polygon':
                        {'x': all_points_x,
                         'y': all_points_y},
                    'name': label}

        seg_list.append(seg_data)

    return seg_list


# # class of SUNRGBD Data
# class SUNRGBDData(object):
#     def __init__(self, K, R_ex, R_tilt, bdb2d, bdb3d, gt3dcorner, imgdepth, imgrgb, seg2d, semantic_seg2d, manhattan_layout,
#                  sequence_name, sequence_id, imgrgb_path, scene_type):
#         self._K = K

#         # R_ex.T is the left-hand camera coordinates -> world coordinates transformation P_world = R_ex*P_camera
#         self._R_ex = R_ex
        
#         # R_tilt is the right-hand camera coordinates  -> world coordinates transformation P_world = R_tilt*P_camera(after transformed to x, z, -y)
#         self._R_tilt = R_tilt

#         self._bdb2d = bdb2d
#         self._bdb3d = bdb3d
#         self._gt3dcorner = gt3dcorner
#         self._imgdepth = imgdepth
#         self._imgrgb = imgrgb
#         self._seg2d = seg2d
#         self._semantic_seg2d = semantic_seg2d
#         self._manhattan_layout = manhattan_layout
#         self._sequence_name = sequence_name
#         self._sequence_id = sequence_id
#         self._height, self._width = np.shape(self._imgrgb)[:2]
#         self._scene_type = scene_type
#         self._imgrgb_path = imgrgb_path

#     def __str__(self):
#         return '[SUNRGBDData] sequence_name: {}, sequence_id: {}'.format(self._sequence_name, self._sequence_id)

#     def __repr__(self):
#         return self.__str__()

#     @property
#     def width(self):
#         return self._width

#     @property
#     def height(self):
#         return self._height

#     @property
#     def K(self):
#         return self._K

#     @property
#     def R_ex(self):
#         return self._R_ex

#     @property
#     def R_tilt(self):
#         return self._R_tilt

#     @property
#     def bdb2d(self):
#         return self._bdb2d

#     @property
#     def bdb3d(self):
#         return self._bdb3d

#     @property
#     def gt3dcorner(self):
#         return self._gt3dcorner

#     @property
#     def imgdepth(self):
#         return self._imgdepth

#     @property
#     def imgrgb(self):
#         return self._imgrgb

#     @property
#     def seg2d(self):
#         return self._seg2d

#     @property
#     def semantic_seg2d(self):
#         return self._semantic_seg2d

#     @property
#     def manhattan_layout(self):
#         return self._manhattan_layout

#     @property
#     def sequence_name(self):
#         return self._sequence_name

#     @property
#     def sequence_id(self):
#         return self._sequence_id

#     @property
#     def imgrgb_path(self):
#         return self._imgrgb_path

#     @property
#     def scene_type(self):
#         return self._scene_type

# def readsunrgbdframe(config, image_name=None, image_id=None, if_return_img_info=False):
#     clean_data_path = config.clean_data_root
#     with open(os.path.join(clean_data_path, 'imagelist.txt'), 'r') as f:
#         image_list = [line.replace('\n', '') for line in f]
#     f.close()
#     if image_name:
#         image_id = image_list.index(image_name) + 1
#     pickle_path = os.path.join(clean_data_path, 'data_all', str(image_id) + '.pickle')
#     with open(pickle_path, 'rb') as f:
#         img_info = pickle.load(f, encoding='latin1')
#         # print(img_info.keys()) # dict_keys(['bdb2d', 'R_ex', 'seg2d_path', 'bdb3d', 'sequence_name', 'gt3dcorner', 'R_tilt', 'imgdepth_path', 'sensor', 'K', 'imgrgb_path'])

#     # change data root manually
#     img_info['imgrgb_path'] = img_info['imgrgb_path'].replace('/home/siyuan/Documents/Dataset/SUNRGBD_ALL', config.data_root)
#     img_info['imgdepth_path'] = img_info['imgdepth_path'].replace('/home/siyuan/Documents/Dataset/SUNRGBD_ALL', config.data_root)
#     img_info['seg2d_path'] = os.path.join(os.path.dirname(os.path.dirname(img_info['imgdepth_path'])), 'annotation2Dfinal', 'index.json')
#     img_info['semantic_seg_path'] = os.path.join(config.data_root, 'SUNRGBD/train_test_labels', "img-{0:06d}.png".format(image_id))
#     # load rgb img
#     img_info['imgrgb'] = np.array(Image.open(img_info['imgrgb_path']))

#     # load depth img
#     imgdepth = np.array(Image.open(img_info['imgdepth_path'])).astype('uint16')
#     imgdepth = (imgdepth >> 3) | (imgdepth << 13)
#     imgdepth = imgdepth.astype('single') / 1000
#     imgdepth[imgdepth > 8] = 8
#     img_info['imgdepth'] = imgdepth

#     if 'gt3dcorner' not in img_info.keys():
#         img_info['gt3dcorner'] = None

#     # load segmentation
#     img_info['seg2d'] = read_seg2d_data(img_info['seg2d_path'])
#     # print(img_info['seg2d_path'])
#     # img_info['seg2d'] = None

#     img_info['manhattan_layout'] = loadmat(os.path.join(sunrgbd_config.data_root, '3dlayout', str(image_id) + '.mat'))['manhattan_layout'].T

#     scene_category_path = os.path.join(config.data_root, img_info['sequence_name'], 'scene.txt')
#     if not os.path.exists(scene_category_path):
#         scene_category = None
#     else:
#         with open(scene_category_path, 'r') as f:
#             scene_category = f.readline()

#     # use updated R_tilt
#     R_tilt = loadmat(os.path.join(sunrgbd_config.data_root, 'updated_rtilt', str(image_id) + '.mat'))['r_tilt']
#     R_ex = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).dot(R_tilt).dot(
#         np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

#     img_info['R_tilt'] = R_tilt
#     img_info['R_ex'] = R_ex

#     data_frame = SUNRGBDData(img_info['K'], img_info['R_ex'], img_info['R_tilt'], img_info['bdb2d'], img_info['bdb3d'],
#                              img_info['gt3dcorner'], img_info['imgdepth'], img_info['imgrgb'], img_info['seg2d'], img_info['semantic_seg_path'],
#                              img_info['manhattan_layout'], img_info['sequence_name'], image_id, img_info['imgrgb_path'], scene_category)

#     if if_return_img_info:
#         return data_frame, img_info
#     else:    
#         return data_frame


# def cvt_R_ex_to_cam_R(R_ex):
#     '''
#     convert SUNRGBD camera R_ex matrix to **transform objects from world system to camera system**
#     both under the 'toward-up-right' system.
#     :return: cam_R matrix
#     '''
#     trans_mat = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
#     return (trans_mat.T).dot(R_ex).dot(trans_mat)

# from scipy.spatial.transform import Rotation as R    
# import random
# from scipy.spatial import ConvexHull
# from utils.utils_misc import magenta, yellow
# import scipy
# from utils.utils_rui import minimum_bounding_rectangle

# def intersection(lst1, lst2): 
#     lst3 = [value for value in lst1 if value in lst2] 
#     return lst3

# def get_layout_bdb_from_corners(layout_t, image_id=-1, id_str='', obj_type='obj', if_fix=True, if_debug=False):
#     '''
#     get coeffs, basis, centroid from corners
#     :param corners: 8x3 numpy array corners of a 3D bounding box
#     [toward, up, right] coordinates
#     :return: bounding box parameters
#     '''
#     assert layout_t.shape == (8, 3)
#     # print('--->layout_t', layout_t, image_id, obj_type)
#     y_max = layout_t[:, 1].max()
#     y_min = layout_t[:, 1].min()
#     points_2d = layout_t[abs(layout_t[:, 1] - y_max) < 1e-5, :]
#     # print('--', layout_t[:, 1], abs(layout_t[:, 1] - y_max), y_max, abs(layout_t[:, 1] - y_max) < 1e-5)
#     assert points_2d.shape[0] == 4, 'Error with %d-%s-%s'%(image_id, id_str, obj_type if obj_type is not None else 'None')
#     if points_2d.shape[0] != 4:
#         return {'if_valid': False}
#     # area_before = Polygon(np.hstack([points_2d[:, 0], points_2d[:, 2]])).area
#     coords_array = np.array([(x[0], x[2]) for x in points_2d])
#     assert coords_array.shape == (4, 2)
#     try:
#         hull = ConvexHull(coords_array)
#         hull_points = coords_array[hull.vertices]
#         area_before = PolygonArea(hull_points)
#     except scipy.spatial.qhull.QhullError:
#         print(magenta('[%s] coords_array with 0 volume; skipped: ')%id_str, coords_array, ) # Skipped think objects!!!!
#         return None
#     red, rec_HW, rec_area = minimum_bounding_rectangle(coords_array)
#     # print('<---layout_t', layout_t, image_id, obj_type)
#     # print(points_2d, np.argsort(points_2d[:, 0]))

#     # points_2d = points_2d[np.argsort(points_2d[:, 0]), :] # will be problematic when box is aligned with axis!!!! https://i.imgur.com/VykapnZ.png
    
#     # # ---> fix1: https://i.imgur.com/KiITp95.png
#     # points_2d_center = np.mean(points_2d, 0, keepdims=True)
#     # points_2d_centered_ori = points_2d - points_2d_center
#     # points_2d_centered = points_2d_centered_ori
#     # sortted = np.sort(points_2d_centered_ori[:, 0])
#     # thres = 1e-3
#     # tried = 0
#     # while abs(sortted[0]-sortted[1])<thres or abs(sortted[1]-sortted[2])<thres or abs(sortted[2]-sortted[3])<thres and (max(sortted)-min(sortted)) > thres:
#     #     small_degree = random.uniform(0, 3)/max(1., 5.-tried)
#     #     small_r = R.from_rotvec(small_degree/180.*np.pi * np.array([0, 1, 0])).as_matrix()
#     #     tried += 1
#     #     points_2d_centered = points_2d_centered_ori @ small_r.T
#     #     print(small_degree, small_r)
#     #     print(points_2d_centered)
#     #     print(points_2d_centered_ori)
#     #     sortted = np.sort(points_2d_centered[:, 0])
#     #     print('->', sortted)
#     #     if tried > 5:
#     #         print(yellow('Tried %d times: %s'%(tried, id_str)))
#     # points_2d = points_2d_centered + points_2d_center
#     # points_2d = points_2d[np.argsort(points_2d[:, 0]), :]
#     # # <--- fix

#     # ---> fix2: https://i.imgur.com/T83f2jH.jpg
#     # if abs(sortted[0]-sortted[1])<thres or abs(sortted[2]-sortted[3])<thres:

#     if if_debug:
#         if_fix = False

#     thres_thin_obj = 2e-3
#     # print('>>>>>>>', id_str, area_before, hull.area)
#     # if xmax-xmin < thres_thin_obj or zmax-zmin < thres_thin_obj:
#     if min(rec_HW) < thres_thin_obj:
#         print(magenta('[%s] very thin (%.5f) obj; skipped: ')%(id_str, min(rec_HW)) ) # Skipped think objects!!!!
#         return None

#     sortted = np.sort(points_2d[:, 0])
#     xmin = points_2d[:, 0].min()
#     xmax = points_2d[:, 0].max()
#     # zmin = points_2d[:, 2].min()
#     # zmax = points_2d[:, 2].max()
#     thres_x = min(thres_thin_obj, (xmax - xmin) / 10.)
#     thes_x_big = thres_x * 5.
#     # thres_z = min(thres_thin_obj, (zmax - zmin) / 10.)
#     # print('=====>', sortted, abs(sortted[0]-sortted[1])<thres, abs(sortted[2]-sortted[3])<thres)
#     manual_idxes_list = []
#     if (abs(sortted[0]-sortted[1])<thres_x or abs(sortted[2]-sortted[3])<thres_x) and if_fix:
#         assert abs(sortted[0]-sortted[1])<thes_x_big and abs(sortted[2]-sortted[3])<thes_x_big
#         left_indexes = np.where(abs(points_2d[:, 0].flatten()-xmin)<thres_x)[0].tolist()
#         right_indexes = np.where(abs(points_2d[:, 0].flatten()-xmax)<thres_x)[0].tolist()
#         # up_indexes = np.where(abs(points_2d[:, 2].flatten()-zmax)<thres_z)[0].tolist()
#         # down_indexes = np.where(abs(points_2d[:, 2].flatten()-zmin)<thres_z)[0].tolist()
#         # print(id_str, left_indexes, right_indexes, up_indexes, down_indexes)
#         # print(points_2d)
#         assert len(left_indexes) == 2, id_str # two on the left
#         assert len(right_indexes) == 2, id_str # two on the right
#         if points_2d[left_indexes[0], 2] < points_2d[left_indexes[1], 2]:
#             manual_idxes_list += [left_indexes[0], left_indexes[1]]
#         else:
#             manual_idxes_list += [left_indexes[1], left_indexes[0]]
#         if points_2d[right_indexes[0], 2] < points_2d[right_indexes[1], 2]:
#             manual_idxes_list += [right_indexes[0], right_indexes[1]]
#         else:
#             manual_idxes_list += [right_indexes[1], right_indexes[0]]
#         # assert len(up_indexes) == 2, id_str # two on the up
#         # assert len(down_indexes) == 2, id_str # two on the down
#         # idx0 = intersection(left_indexes, down_indexes)
#         # idx1 = intersection(left_indexes, up_indexes)
#         # idx2 = intersection(right_indexes, down_indexes)
#         # idx3 = intersection(right_indexes, up_indexes)
#         # assert len(idx0) == len(idx1) == len(idx2) == len(idx3) == 1
#         # manual_idxes_list = idx0 + idx1 + idx2 + idx3
#         points_2d = points_2d[manual_idxes_list, :]
#         sortted = np.sort(points_2d[:, 0])
#     else:
#         points_2d = points_2d[np.argsort(points_2d[:, 0]), :]
        
#     # <--- fix

#     vector1 = points_2d[3] - points_2d[1]
#     vector2 = points_2d[1] - points_2d[0]

#     coeff1 = np.linalg.norm(vector1)
#     coeff2 = np.linalg.norm(vector2)

#     vector1 = normalize_point(vector1)
#     vector2 = np.cross(vector1, [0, 1, 0]) # https://i.imgur.com/l7ltQ8u.png

#     centroid = np.array(
#         [points_2d[0, 0] + points_2d[3, 0], float(y_max) + float(y_min), points_2d[0, 2] + points_2d[3, 2]]) * 0.5

#     basis = np.array([vector1, [0, 1, 0], vector2])
#     coeffs = np.array([coeff1, y_max-y_min, coeff2]) * 0.5
#     area_after = coeffs[0] * coeffs[2] * 4
#     area_ratio = abs(area_before - area_after) / (area_before + 1e-5)
#     if area_ratio > 0.05:
#         print(area_before / area_after, area_before, '/', area_after)
#         print(magenta('Small area Error!' + id_str))
#         print(layout_t)
#         # if area_after < 1e-3:
#         #     assert False
#     if not if_debug:
#         assert area_ratio < 0.05, id_str
#     assert np.linalg.det(basis) > 0.
#     bdb = {'centroid':centroid, 'basis':basis, 'coeffs':coeffs, 'if_valid': True, 'basis_origin': points_2d[1]}

#     return bdb


# def process_layout(layout, if_swap_axis=True, if_return_bdb=True, image_id=None, obj_type='layout'):
#     '''
#     transform sunrgbd layout to toward-up-right form.
#     :param layout: sunrgbd layout
#     :return: toward-up-right form.
#     '''
#     if if_swap_axis:
#         trans_mat = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
#         # layout_t = (trans_mat.dot(layout.T)).T # == layout @ trans_mat.T
#         layout_t = layout @ trans_mat.T # [x1, x2, x3] -> [x2, x3, x1], just swapping axes ([right x, toward y, up z] -> [toward, up right])
#     else:
#         layout_t = layout
#     if if_return_bdb:
#         bdb = get_layout_bdb_from_corners(layout_t, image_id=image_id, obj_type=obj_type)
#         return bdb, layout_t
#     else:
#         return layout_t


# def check_bdb(bdb2d, m, n):
#     """
#         Check valid a bounding box is valid

#         Parameters
#         ----------
#         bdb2d: dict
#             Keys: {'x1', 'x2', 'y1', 'y2'}
#             The (x1, y1) position is at the top left corner,
#             the (x2, y2) position is at the bottom right corner
#         m: int
#             width
#         n: int
#             height

#         Returns
#         -------
#         valid: bool
#     """
#     if bdb2d['x1'] >= bdb2d['x2'] or bdb2d['y1'] >= bdb2d['y2'] or bdb2d['x1'] > m or bdb2d['y1'] > n:
#         # print(bdb2d, m, n)
#         return False
#     else:
#         return True


# def check_bdb2d(bdb2ds, WH_tuple, if_clip=False):
#     W, H = WH_tuple
#     result = []
#     valid_idxes = []

#     for idx, bdb2d in enumerate(bdb2ds):
#         if if_clip:
#             # H, W = img_shape[1] - 1, img_shape[0] - 1
#             # print('--->', bdb2d)
#             bdb2d['x1'] = np.clip(bdb2d['x1'], 0., W-1)
#             bdb2d['x2'] = np.clip(bdb2d['x2'], 0., W-1)
#             bdb2d['y1'] = np.clip(bdb2d['y1'], 0., H-1)
#             bdb2d['y2'] = np.clip(bdb2d['y2'], 0., H-1)
#             # print(bdb2d, '---->')
#         if check_bdb(bdb2d, W, H):
#             result.append(bdb2d)
#             valid_idxes.append(idx)
#         else:
#             print('ground truth not valid')
#     return result, valid_idxes


# def find_close_name(name, label_list):
#     '''
#     find a close name from label list
#     :param name: input name
#     :param label_list: name dictionary
#     :return: close name.
#     '''
#     leve = {}
#     for label in label_list:
#         leve[label] = jf.jaro_distance(name, label)

#     return max(leve, key=leve.get)


# def cvt2nyu37class_map(inst_map, mapping):

#     class_map = np.zeros_like(inst_map)

#     for key, value in mapping.items():
#         class_map[inst_map == key] = value

#     return class_map

# def get_inst_map(seg2d_data, cls_map):
#     '''
#     get 2D instance map from segmented polygons.
#     :param seg2d_data: polygon data for each segmented object.
#     :param cls_map: semantic cls maps.
#     :return: 2D instance map with NYU37 labels.
#     '''

#     inst_map = np.zeros_like(cls_map, dtype=np.uint8)
#     inst_cls = {}

#     for inst_id, inst in enumerate(seg2d_data):
#         mask = np.zeros_like(cls_map)
#         if len(inst['polygon']['x']) != 0:
#             cv2.fillConvexPoly(mask, np.vstack([inst['polygon']['x'], inst['polygon']['y']]).T, 1)
#         labels, counts = np.unique(cls_map[np.nonzero(mask)], return_counts=True)
#         if len(counts) == 0 :
#             continue
#         inst_cls[inst_id + 1] = labels[counts.argmax()]
#         cv2.fillConvexPoly(inst_map, np.vstack([inst['polygon']['x'], inst['polygon']['y']]).T, inst_id + 1)

#     return inst_map, inst_cls

# def get_campact_layout(layout, depth_map, cam_K, cam_R, bdb3ds):

#     # get 3d points cloud from depth map
#     u, v = np.meshgrid(range(depth_map.shape[1]), range(depth_map.shape[0]))
#     u = u.reshape([1, -1])[0]
#     v = v.reshape([1, -1])[0]

#     z_cam = depth_map[v, u]

#     # remove zeros
#     non_zero_indices = np.argwhere(z_cam).T[0]
#     z_cam = z_cam[non_zero_indices]
#     u = u[non_zero_indices]
#     v = v[non_zero_indices]

#     # calculate coordinates
#     x_cam = (u - cam_K[0][2]) * z_cam / cam_K[0][0]
#     y_cam = (v - cam_K[1][2]) * z_cam / cam_K[1][1]

#     # transform to toward-up-right coordinate system
#     x3 = z_cam
#     y3 = -y_cam
#     z3 = x_cam

#     # transform from camera system to layout system
#     points_cam = np.vstack([x3, y3, z3]).T
#     points_cloud = points_cam.dot(cam_R.T).dot(layout['basis'].T)

#     # layout corners in layout system
#     layout_corners = get_corners_of_bb3d_no_index(layout['basis'], layout['coeffs'], layout['centroid']).dot(layout['basis'].T)

#     # instance corners in layout system
#     instance_corners = []
#     for bdb3d in bdb3ds:
#         instance_corners.append(get_corners_of_bb3d_no_index(bdb3d['basis'], bdb3d['coeffs'], bdb3d['centroid']).dot(layout['basis'].T))

#     if instance_corners:

#         instance_corners = np.vstack(instance_corners)

#         # scope
#         x_min = min(points_cloud[:, 0].min(), instance_corners[:, 0].min())
#         x_max = max(min(layout_corners[:, 0].max(), points_cloud[:, 0].max()), instance_corners[:, 0].max())

#         y_min = min(max(points_cloud[:, 1].min(), layout_corners[:, 1].min()), instance_corners[:, 1].min())
#         y_max = y_min + 3.

#         z_min = min(max(layout_corners[:, 2].min(), points_cloud[:, 2].min()), instance_corners[:, 2].min())
#         z_max = max(min(layout_corners[:, 2].max(), points_cloud[:, 2].max()), instance_corners[:, 2].max())

#     else:
#         # scope
#         x_min = points_cloud[:, 0].min()
#         x_max = min(layout_corners[:, 0].max(), points_cloud[:, 0].max())

#         y_min = max(points_cloud[:, 1].min(), layout_corners[:, 1].min())
#         y_max = y_min + 3.

#         z_min = max(layout_corners[:, 2].min(), points_cloud[:, 2].min())
#         z_max = min(layout_corners[:, 2].max(), points_cloud[:, 2].max())

#     new_layout_centroid = np.array([(x_min + x_max)/2., (y_min + y_max)/2., (z_min + z_max)/2.])
#     new_layout_coeffs = np.array([(x_max - x_min)/2., (y_max - y_min)/2., (z_max - z_min)/2.])

#     new_layout = deepcopy(layout)

#     new_layout['centroid'] = new_layout_centroid.dot(layout['basis'])
#     new_layout['coeffs'] = new_layout_coeffs

#     return new_layout

# def get_NYU37_class_id(names):
#     '''
#     get the NYU class id for each class name.
#     :param names: class names
#     :return: nyu id.
#     '''

#     Name_6585 = class_mapping.Name_6585.values.astype('str')

#     nyu37class_dict = {}

#     for inst_id, name in enumerate(names):

#         # process name
#         name = name.lower()
#         name = ''.join(i for i in name if not i.isdigit())

#         # match name in class_mapping
#         name = name if name in Name_6585 else find_close_name(name, Name_6585)
#         nyu37class_dict[inst_id + 1] = class_mapping[class_mapping.Name_6585 == name].Label_37.item()

#     return nyu37class_dict


# def process_bdb2d(bdb2ds, WH_tuple):

#     W, H = WH_tuple

#     bdb2ds_t_list = []

#     for bdb2d in bdb2ds:

#         bdb2ds_t = {}

#         for key in bdb2d:
#             if key not in ['x1', 'x2', 'y1', 'y2']:
#                 # if type(key) is not str:
#                 bdb2ds_t[key] = bdb2d[key]

#         if 'class_id' in bdb2d.keys():
#             class_id = bdb2d['class_id']
#             cat_name = bdb2d['cat_name']
#         else:
#             class_id = get_NYU37_class_id([bdb2d['classname']])[1]
#             cat_name = bdb2d['classname']

#         bdb2ds_t['class_id'] = class_id
#         bdb2ds_t['cat_name'] = cat_name
#         bdb2ds_t['x1'] = max(bdb2d['x1'], 0)
#         bdb2ds_t['y1'] = max(bdb2d['y1'], 0)
#         bdb2ds_t['x2'] = min(bdb2d['x2'], W - 1)
#         bdb2ds_t['y2'] = min(bdb2d['y2'], H - 1)
        

#         bdb2ds_t_list.append(bdb2ds_t)

#     return bdb2ds_t_list


# def process_msk(bdb2ds, cls_masks, seg2d, flip_seg=False):
#     '''
#     get instance masks from semantic masks
#     :param bdb2ds: instance bounding boxes
#     :param cls_masks: semantic masks
#     :return: instance masks with each entry as instance id.
#     '''
#     # recover the NYU 37 class label for each object
#     inst_cls = []
#     inst_masks = []

#     if not flip_seg:
#         for inst_id, inst in enumerate(seg2d):
#             if ('polygon' not in inst) or ('x' not in inst['polygon']) or ('y' not in inst['polygon']) or (
#             not inst['polygon']['x']) or (not inst['polygon']['y']):
#                 continue

#             mask = np.zeros_like(cls_masks)
#             cv2.fillConvexPoly(mask, np.vstack([inst['polygon']['x'], inst['polygon']['y']]).T, 1)
#             labels, counts = np.unique(cls_masks[np.nonzero(mask)], return_counts=True)
#             if len(counts) == 0 :
#                 continue
#             inst_cls.append(labels[counts.argmax()])
#             inst_masks.append(mask)
#     else:

#         for inst_id, inst in enumerate(seg2d):
#             if ('polygon' not in inst) or ('x' not in inst['polygon']) or ('y' not in inst['polygon']) or len(
#                     inst['polygon']['x']) == 0 or len(inst['polygon']['y']) == 0:
#                 continue

#             mask = np.zeros_like(cls_masks)
#             cv2.fillConvexPoly(mask,
#                                np.vstack([mask.shape[1] - 1 - np.array(inst['polygon']['x']), inst['polygon']['y']]).T,
#                                1)
#             labels, counts = np.unique(cls_masks[np.nonzero(mask)], return_counts=True)
#             if len(counts) == 0 :
#                 continue
#             inst_cls.append(labels[counts.argmax()])
#             inst_masks.append(mask)

#     inst_masks = np.stack(inst_masks)

#     target_inst_masks = []
#     for inst_id, bdb2d in enumerate(bdb2ds):
#         candidate_inst_ids = [idx for idx, cls in enumerate(inst_cls) if cls == bdb2d['class_id']]

#         if not candidate_inst_ids:
#             target_inst_masks.append(None)
#             continue

#         candidate_inst_masks = inst_masks[candidate_inst_ids]

#         n_pixel_for_each_inst = np.sum(candidate_inst_masks.reshape(candidate_inst_masks.shape[0], -1), axis=1)
#         in_box_inst_masks = candidate_inst_masks[:, bdb2d['y1']:bdb2d['y2'] + 1, bdb2d['x1']:bdb2d['x2'] + 1]
#         n_in_box_pixel_for_each_inst = np.sum(in_box_inst_masks.reshape(in_box_inst_masks.shape[0], -1), axis=1)
#         in_box_ratio = n_in_box_pixel_for_each_inst/n_pixel_for_each_inst

#         if True not in (in_box_ratio >= 0.8):
#             target_inst_masks.append(None)
#             continue

#         target_inst_mask = candidate_inst_masks[in_box_ratio >= 0.8].sum(0).astype(np.bool_)
#         locs = np.argwhere(target_inst_mask)
#         y1, x1 = locs.min(0)
#         y2, x2 = locs.max(0)
#         target_inst_mask = {'msk_bdb': [x1, y1, x2, y2], 'msk': target_inst_mask[y1:y2 + 1, x1:x2 + 1], 'class_id':bdb2d['class_id']}
#         target_inst_masks.append(target_inst_mask)

#     return target_inst_masks


# def process_bdb3d(bdb3ds, if_trans_towardupright=True, piggyback_center_axis_vectors_list=[]):
#     '''
#     transform sunrgbd layout to toward-up-right form in world system.
#     :param layout: sunrgbd layout
#     :return: toward-up-right form.
#     '''
#     if if_trans_towardupright:
#         trans_mat = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
#     else:
#         trans_mat = np.eye(3)
#     bdb3ds_t = []
#     valid_idxes = []
#     for idx, bdb3d in enumerate(bdb3ds):
#         # if bdb3d['if_valid'] == False:
#         #     bdb3ds_t.append({'if_valid': False})
#             # continue
#         # print('-----', bdb3d['centroid'], bdb3d['coeffs'], bdb3d['basis'])
#         centroid = bdb3d['centroid']
#         if centroid.shape == (1,):
#             centroid = centroid[0]
#         centroid = trans_mat.dot(centroid.reshape(3,)) # sunrgbd: [x1, x2, x3] -> [x2, x3, x1], just swapping axes
#         coeffs = bdb3d['coeffs'].reshape(3,)
#         basis = bdb3d['basis'].astype('float32')
#         assert basis.shape == (3, 3)
#         # print(centroid, coeffs, basis, trans_mat)
#         vectors = trans_mat.dot((trans_mat.dot((np.diag(coeffs).dot(basis)).T)).T)

#         # let z-axis face forward (consistent with suncg data.)
#         vectors = np.array([vectors[2], vectors[1], -vectors[0]])
#         vectors[0] = vectors[0] if np.linalg.det(vectors)>0. else -vectors[0]

#         bdb3d_t = {'if_valid': True}
#         bdb3d_t['coeffs'] = np.linalg.norm(vectors, axis=1)
#         bdb3d_t['basis'] = np.array([normalize_point(vector) for vector in vectors])
#         if np.linalg.det(vectors)<=0.:
#             assert False
#             continue
#         bdb3d_t['centroid'] = centroid
#         if 'class_id' in bdb3d:
#             bdb3d_t['class_id'] = bdb3d['class_id']
#             bdb3d_t['cat_name'] = bdb3d['cat_name']
#         else:
#             bdb3d_t['class_id'] = get_NYU37_class_id(bdb3d['classname'])[1]
#         if 'random_id' in bdb3d:
#             bdb3d_t['random_id'] = bdb3d['random_id']

#         bdb3ds_t.append(bdb3d_t)
#         valid_idxes.append(idx)

#         # print('----->', bdb3d['centroid'], bdb3d['coeffs'], bdb3d['basis'])

#     if len(piggyback_center_axis_vectors_list) != 0:
#         return bdb3ds_t, valid_idxes, [(trans_mat.dot(x[0].reshape(3,)), trans_mat.dot(x[1].reshape(3,))) for x in piggyback_center_axis_vectors_list]
#     else:
#         return bdb3ds_t, valid_idxes



# # def process_bdb3d_old(bdb3ds):
# #     '''
# #     transform sunrgbd layout to toward-up-right form in world system.
# #     :param layout: sunrgbd layout
# #     :return: toward-up-right form.
# #     '''
# #     trans_mat = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
# #     bdb3ds_t = []
# #     for bdb3d in bdb3ds:
# #         # print('-----', bdb3d['centroid'], bdb3d['coeffs'], bdb3d['basis'], bdb3d['centroid'][0].shape, bdb3d['coeffs'][0].shape)
# #         centroid = trans_mat.dot(bdb3d['centroid'][0]) # [x1, x2, x3] -> [x2, x3, x1], just swapping axes
# #         coeffs = bdb3d['coeffs'][0]
# #         basis = bdb3d['basis'].astype('float32')
# #         vectors = trans_mat.dot((trans_mat.dot((np.diag(coeffs).dot(basis)).T)).T)

# #         # let z-axis face forward (consistent with suncg data.)
# #         vectors = np.array([vectors[2], vectors[1], -vectors[0]])
# #         vectors[0] = vectors[0] if np.linalg.det(vectors)>0. else -vectors[0]

# #         bdb3d_t = {}
# #         bdb3d_t['coeffs'] = np.linalg.norm(vectors, axis=1)
# #         bdb3d_t['basis'] = np.array([normalize_point(vector) for vector in vectors])
# #         if np.linalg.det(vectors)<=0.:
# #             continue
# #         bdb3d_t['centroid'] = centroid
# #         bdb3d_t['class_id'] = get_NYU37_class_id(bdb3d['classname'])[1]
# #         bdb3ds_t.append(bdb3d_t)
# #         print(centroid.shape)

# #     return bdb3ds_t


# def transform_to_world(layout, bdb3ds, cam_R, world_R):
#     '''
#     transform scene to global world system
#     :param layout_3D:
#     :param bdb3ds_ws:
#     :param cam_R:
#     :param world_R:
#     :return:
#     '''
#     new_layout = deepcopy(layout)
#     # print(layout['centroid'], world_R)
#     # print(layout['centroid'].shape, world_R.shape)    
#     new_layout['centroid'] = layout['centroid'].dot(world_R)  # layout centroid in world system
#     new_layout['basis'] = layout['basis'].dot(world_R)  # layout vectors in world system
#     # new_layout['basis_origin'] = layout['basis_origin'].dot(world_R)

#     new_cam_R = (world_R.T).dot(cam_R) # offseting the yaw component in cam_R with world_R; now new_cam_R is yaw_free

#     new_bdb3ds = []
#     for bdb3d in bdb3ds:
#         new_bdb3d = deepcopy(bdb3d)
#         new_bdb3d['centroid'] = bdb3d['centroid'].dot(world_R)
#         new_bdb3d['basis'] = bdb3d['basis'].dot(world_R)

#         if 'emitter_prop' in bdb3d and 'light_world_total3d_centeraxis' in bdb3d['emitter_prop']:
#             new_bdb3d['emitter_prop']['light_world_total3d_centeraxis'] = (new_bdb3d['emitter_prop']['light_world_total3d_centeraxis'][0].dot(world_R), \
#                 new_bdb3d['emitter_prop']['light_world_total3d_centeraxis'][1].dot(world_R))

#         new_bdb3ds.append(new_bdb3d)

#     return new_layout, new_bdb3ds, new_cam_R


# def flip_layout(layout, cam_R, cam_R_flip):
#     '''
#     transform and flip sunrgbd layout to toward-up-right form.
#     :param layout: sunrgbd layout
#     :return: toward-up-right form.
#     '''

#     # layout is the layout coordinates in world system (toward-up-right form).
#     centroid_flip = layout['centroid'].dot(cam_R)  # layout centroid in camera system
#     centroid_flip[2] = -1 * centroid_flip[2]    # flip right-coordinate values
#     centroid_flip = centroid_flip.dot(cam_R_flip.T)  # transform back to world system

#     vectors_flip = np.diag(layout['coeffs']).dot(layout['basis']).dot(cam_R) # layout vectors in camera system
#     vectors_flip[:,2] = -1 * vectors_flip[:,2] # flip right-coordinate values
#     vectors_flip = vectors_flip.dot(cam_R_flip.T) # transform back to world system

#     coeffs_flip = np.linalg.norm(vectors_flip, axis=1)
#     basis_flip = np.array([normalize_point(vector) for vector in vectors_flip])

#     basis_flip[2, :] = basis_flip[2, :] if np.linalg.det(basis_flip)>0 else -basis_flip[2, :]

#     bdb_flip = {}
#     bdb_flip['basis'] = basis_flip
#     bdb_flip['coeffs'] = coeffs_flip
#     bdb_flip['centroid'] = centroid_flip

#     return bdb_flip

# def flip_bdb2d(bdb2ds, im_width):

#     bdb2ds_flip = deepcopy(bdb2ds)

#     for bdb_idx, bdb2d in enumerate(bdb2ds):
#         bdb2ds_flip[bdb_idx]['x1'] = im_width - 1 - bdb2d['x2']
#         bdb2ds_flip[bdb_idx]['x2'] = im_width - 1 - bdb2d['x1']

#     return bdb2ds_flip


# def flip_bdb3d(bdb3ds, cam_R, cam_R_flip):

#     bdb3ds_flip = deepcopy(bdb3ds)

#     for bdb_idx, bdb3d in enumerate(bdb3ds):
#         centroid_flip = bdb3d['centroid'].dot(cam_R) # transform bdb centroid to camera system
#         centroid_flip[2] = -1 * centroid_flip[2] # flip right-coordinate
#         centroid_flip = centroid_flip.dot(cam_R_flip.T) # transform back to world system

#         vectors_flip = np.diag(bdb3d['coeffs']).dot(bdb3d['basis']).dot(cam_R) # transform vectors to camera system
#         vectors_flip[:, 2] = -1 * vectors_flip[:, 2] # flip right-coordinate
#         vectors_flip = vectors_flip.dot(cam_R_flip.T) # transform back to world system

#         coeffs_flip = np.linalg.norm(vectors_flip, axis=1)
#         basis_flip = np.array([normalize_point(vector) for vector in vectors_flip])

#         # keep the basis_flip[2,:] vector, because it stands for the forward direction of an object.
#         basis_flip[0, :] = basis_flip[0, :] if np.linalg.det(basis_flip) > 0 else -basis_flip[0, :]

#         bdb3ds_flip[bdb_idx]['basis'] = basis_flip
#         bdb3ds_flip[bdb_idx]['coeffs'] = coeffs_flip
#         bdb3ds_flip[bdb_idx]['centroid'] = centroid_flip

#     return bdb3ds_flip

# def process_sunrgbd_frame(sample, flip=False):
    '''
    Read SUNRGBD frame and transform all 3D data to 'toward-up-right' layout system.
    :param sample: SUNRGBD frame
    :return:
    '''
    # TODO: define global coordinate system
    if not flip:
        cam_K = sample.K
        cam_R = cvt_R_ex_to_cam_R(sample.R_ex) # camera_rotation matrix in world system

        # define a world system
        world_R, _ = get_world_R(cam_R)

        layout, _ = process_layout(sample.manhattan_layout, if_swap_axis=True) # layout bbox in world system

        centroid = layout['centroid']
        vectors = np.diag(layout['coeffs']).dot(layout['basis'])

        # Set all points relative to layout orientation. (i.e. let layout orientation to be the world system.)
        # The forward direction (x-axis) of layout orientation should point toward camera forward direction.
        layout_3D = get_layout_info({'centroid': centroid, 'vectors': vectors}, cam_R[:, 0])

        bdb2ds = process_bdb2d(check_bdb2d(sample.bdb2d, (sample.width, sample.height)), (sample.width, sample.height))
        masks = np.array(Image.open(sample.semantic_seg2d))
        masks = process_msk(bdb2ds, masks, sample.seg2d, flip_seg=False)
        bdb3ds_ws, _ = process_bdb3d(sample.bdb3d) # bdb3d in old world system
        assert len(bdb3ds_ws) == len(bdb2ds)

        # transform everything to world system
        # print('---world_R', world_R)
        layout_3D, bdb3ds_ws, cam_R = transform_to_world(layout_3D, bdb3ds_ws, cam_R, world_R)

        instance_info_list = {} # bdb2d and bdb3d in layout system

        instance_info_list['bdb2d'] = bdb2ds
        instance_info_list['bdb3d'] = bdb3ds_ws
        instance_info_list['inst_masks'] = masks
        # layout_3D = get_campact_layout(layout_3D, sample.imgdepth, cam_K, cam_R, bdb3ds_ws)

        frame = SUNRGBD_DATA(cam_K, cam_R, sample.scene_type, sample.imgrgb, sample.imgdepth, layout_3D, sample.sequence_id,
                           sample.sequence_name, sample.imgrgb_path, instance_info_list)
    else:
        assert False, 'Not implemented... yet!'
        img_shape = sample.imgrgb.shape[:2]

        cam_K_flip = deepcopy(sample.K)
        cam_K_flip[0][2] = img_shape[1] - cam_K_flip[0][2] # flip cam_K

        # camera vectors in world system.
        cam_R = cvt_R_ex_to_cam_R(sample.R_ex) # camera_rotation matrix in world system
        _, pitch, roll = yaw_pitch_roll_from_R(cam_R)
        # flip camera R
        cam_R_flip = R_from_yaw_pitch_roll(0, pitch, -roll)

        # get ordinary layout first in world system.
        layout = process_layout(sample.manhattan_layout)  # layout bbox in world system

        centroid = layout['centroid']
        vectors = np.diag(layout['coeffs']).dot(layout['basis'])

        # The forward direction (x-axis) of layout orientation should point toward camera forward direction.
        layout_3D = get_layout_info({'centroid': centroid, 'vectors': vectors}, cam_R[:, 0])

        # flip layout (we now need to horienzontally flip layout in camera system first and transform it back to world system.)
        layout_3D_flip = flip_layout(layout_3D, cam_R, cam_R_flip) # flipped layout bbox in world system

        # Set all points relative to layout orientation. (i.e. let layout orientation to be the world system.)
        bdb2ds = process_bdb2d(check_bdb2d(sample.bdb2d, sample.imgrgb.shape), sample.imgrgb.shape)
        bdb2ds_flip = flip_bdb2d(bdb2ds, sample.imgrgb.shape[1])
        masks = np.array(Image.open(sample.semantic_seg2d).transpose(Image.FLIP_LEFT_RIGHT))
        masks = process_msk(bdb2ds_flip, masks, sample.seg2d, flip_seg=True)
        bdb3ds_ws, _ = process_bdb3d(sample.bdb3d)  # bdb3d in world system
        bdb3ds_ws_flip, _ = flip_bdb3d(bdb3ds_ws, cam_R, cam_R_flip)
        assert len(bdb3ds_ws) == len(bdb2ds)
        assert len(bdb3ds_ws_flip) == len(bdb2ds_flip)

        instance_info_list = {}  # bdb2d and bdb3d in layout system

        instance_info_list['bdb2d'] = bdb2ds_flip
        instance_info_list['bdb3d'] = bdb3ds_ws_flip
        instance_info_list['inst_masks'] = masks
        # # get compact layout
        # depth_img_flip = np.array(Image.fromarray(sample.imgdepth).transpose(Image.FLIP_LEFT_RIGHT))
        # layout_3D_flip = get_campact_layout(layout_3D_flip, depth_img_flip, cam_K_flip, cam_R_flip, bdb3ds_ws_flip)

        # flip image in the end.
        rgb_img = np.array(Image.fromarray(sample.imgrgb).transpose(Image.FLIP_LEFT_RIGHT))
        depth_map = np.array(Image.fromarray(sample.imgdepth).transpose(Image.FLIP_LEFT_RIGHT))
        frame = SUNRGBD_DATA(cam_K_flip, cam_R_flip, sample.scene_type, rgb_img, depth_map, layout_3D_flip,
                             sample.sequence_id, sample.sequence_name, sample.imgrgb_path, instance_info_list)

    return frame