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
# import vtk
# from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
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

# import vtk
# from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
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

def format_mesh(obj_files, bboxes, if_use_vtk=True, validate_classids=False):

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
            object.Update()

            # get points from object
            polydata = object.GetOutput()
            # read points using vtk_to_numpy
            points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)
        else:
            points, faces = loadMesh(obj_file)

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




class Box(Scene3D):

    def __init__(self, img_map, depth_map, cam_K, gt_cam_R, pre_cam_R, gt_layout, pre_layout, gt_boxes, pre_boxes, gt_meshes, pre_meshes, \
                    data_type, output_mesh=None, \
                    opt=None, dataset='OR', if_hide_invalid_cats=True, description='', if_mute_print=False, OR=None, \
                    cam_fromt_axis_id=0, emitters_obj_list=None, \
                    emitter2wall_assign_info_list=None, emitter_cls_prob_PRED=None, emitter_cls_prob_GT=None, \
                    cell_info_grid_GT=None, cell_info_grid_PRED=None, \
                    grid_size=4, transform_R=None, transform_t=None, paths={}, pickle_id=-1, \
                    gt_boxes_valid_mask_extra=None, pre_boxes_valid_mask_extra=None):
        super(Scene3D, self).__init__()
        self.opt = opt

        self._cam_K = cam_K
        self.gt_cam_R = gt_cam_R
        # self._cam_R = gt_cam_R
        self.pre_cam_R = pre_cam_R
        self.gt_layout = gt_layout
        self.pre_layout = pre_layout
        self.gt_boxes = gt_boxes
        self.pre_boxes = pre_boxes
        self.gt_meshes = gt_meshes
        self.pre_meshes = pre_meshes

        self.gt_boxes_valid_mask_extra = gt_boxes_valid_mask_extra
        self.pre_boxes_valid_mask_extra = pre_boxes_valid_mask_extra

        self.mode = data_type
        assert self.mode in ['prediction', 'GT', 'both']
        self._img_map = img_map
        self._depth_map = depth_map
        if self.mode == 'prediction':
            self.output_mesh = output_mesh
        
        self.dataset = dataset

        self.classes = None
        self.valid_class_ids = None
        self.pickle_id = pickle_id

        assert self.dataset.lower() in ['or', 'sunrgbd'], 'Unsupported dataset in Box class!'
        if self.dataset.lower() == 'or':
            assert OR is not None
            self.classes = OR4XCLASSES_dict[OR]
            self.valid_class_ids = RECON_3D_CLS_OR_dict[OR]
            self.color_file = opt.cfg.PATH.OR4X_mapping_catInt_to_RGB[0] if opt is not None else paths['color_file']
            with (open(self.color_file, "rb")) as f:
                OR4X_mapping_catInt_to_RGB_light = pickle.load(f)
            self.color_palette = OR4X_mapping_catInt_to_RGB_light[OR]
        elif self.dataset.lower() == 'sunrgbd':
            self.classes = NYU40CLASSES
            self.valid_class_ids = RECON_3D_CLS
            self.color_palette = nyu_color_palette

        self.if_hide_invalid_cats = if_hide_invalid_cats

        self.description = description
        self.if_mute_print = if_mute_print

        self.cam_fromt_axis_id = cam_fromt_axis_id

        # # ------ 3D vis
        # self.gt_layout = gt_layout
        
        # ------ emitters
        self.grid_size = grid_size
        self.emitters_obj_list = emitters_obj_list
        self.emitter2wall_assign_info_list = emitter2wall_assign_info_list
        self.emitter_cls_prob_PRED = emitter_cls_prob_PRED
        self.emitter_cls_prob_GT = emitter_cls_prob_GT

        if self.emitter_cls_prob_PRED is not None:
            self.emitter_cls_prob_PRED = self.emitter_cls_prob_PRED.reshape((6, -1))
            assert self.emitter_cls_prob_PRED.shape[1] in [grid_size**2, grid_size**2+1]
        if self.emitter_cls_prob_GT is not None:
            self.emitter_cls_prob_GT = self.emitter_cls_prob_GT.reshape((6, -1))
            assert self.emitter_cls_prob_GT.shape[1] in [grid_size**2, grid_size**2+1]

        self.cell_info_grid_GT = cell_info_grid_GT
        self.cell_info_grid_PRED = cell_info_grid_PRED
        if self.cell_info_grid_GT is not None:
            assert len(self.cell_info_grid_GT) <= 6 * self.grid_size * self.grid_size
        if self.cell_info_grid_PRED is not None:
            assert len(self.cell_info_grid_PRED) <= 6 * self.grid_size * self.grid_size, 'Wrong length of self.cell_info_grid_PRED: %d VS %d'%(len(self.cell_info_grid_PRED), 6 * self.grid_size * self.grid_size)

        # ------ other transformations
        self.transform_R = transform_R
        self.transform_t = transform_t

        # --- check objects
        if self.gt_boxes is not None:
            all_lengths = [len(self.gt_boxes[key]) for key in self.gt_boxes]
            # for key in self.gt_boxes:
            #     print(key, len(self.gt_boxes[key]))
            assert len(list(set(all_lengths)))==1
            self.gt_boxes_num = all_lengths[0]
            if self.gt_boxes_valid_mask_extra is not None:
                assert len(self.gt_boxes_valid_mask_extra) == self.gt_boxes_num
                self.gt_boxes['if_valid'] = [x[0] and x[1] for x in zip(self.gt_boxes['if_valid'], self.gt_boxes_valid_mask_extra)]

        if self.pre_boxes is not None:
            all_lengths = [len(self.pre_boxes[key]) for key in self.pre_boxes]
            assert len(list(set(all_lengths)))==1
            self.pre_boxes_num = all_lengths[0]
            if self.pre_boxes_valid_mask_extra is not None:
                assert len(self.pre_boxes_valid_mask_extra) == self.pre_boxes_num
                self.pre_boxes['if_valid'] = [x[0] and x[1] for x in zip(self.pre_boxes['if_valid'], self.pre_boxes_valid_mask_extra)]

    def set_cam_K(self, cam_K):
        self.cam_k = cam_K

    def get_valid_class_ids(self):
        return [(x, self.classes[x]) for x in self.valid_class_ids]

    
    def save_img(self, save_path):
        img_map = Image.fromarray(self.img_map[:])
        img_map.save(str(save_path))

    def draw_projected_depth(self, type = 'prediction', return_plt = False, if_save = True, save_path='', if_vis=True, if_use_plt=False, fig_or_ax=None, cam_K_override=None, if_original_lim=True, override_img=None):
        if cam_K_override is not None:
            cam_K = cam_K_override
        else:
            cam_K = self.cam_K
            
        if override_img is None:
            im_uint8 = self.img_map[:]
        else:
            im_uint8 = override_img
        im_height, im_width = im_uint8.shape[:2]

        # if fig_or_ax is None:
        #     fig_2d = plt.figure(figsize=(15, 8))
        #     ax_2d = fig_2d.gca()
        # else:
        #     ax_2d = fig_or_ax
        # ax_2d.imshow(im_uint8)
        # if not if_original_lim:
        #     ax_2d.set_xlim([-im_width*0.5, im_width*1.5])
        #     ax_2d.set_ylim([im_height*1.5, -im_height*0.5])
        # else:
        #     ax_2d.set_xlim([0, im_width])
        #     ax_2d.set_ylim([im_height, 0])

        if type == 'prediction':
            boxes = [self.pre_layout]
            cam_Rs = [self.pre_cam_R]
            colors = [[1., 0., 0.]]
            line_widths = [5]
            linestyles = ['-']
        elif type == 'both':
            boxes = [self.pre_layout, self.gt_layout]
            cam_Rs = [self.pre_cam_R, self.gt_cam_R]
            colors = [[1., 0., 0.], [0., 0., 1.]]
            line_widths = [5, 3]
            linestyles = ['-', '--']
        elif type == 'GT':
            boxes = [self.gt_layout]
            cam_Rs = [self.gt_cam_R]
            colors = [[0., 0., 1.]]
            line_widths = [3]
            linestyles = ['--']
        else:
            assert False, 'not valid Type!'

        for box, cam_R, color, line_width, linestyle in zip(boxes, cam_Rs, colors, line_widths, linestyles):
            if box is None:
                print('[draw_projected_layout] box is None for vis type: %s; skipped'%type)
                continue
            if not isinstance(box, dict):
                box = format_layout(box)
            coeffs, centroid, class_id, basis  = box['coeffs'], box['centroid'], -1, box['basis']
            bdb3d_corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)

            cam_dict = {'origin': np.array([0., 0., 0.]), 'cam_axes': cam_R.T, 'f_x': cam_K[0][0], 'f_y': cam_K[1][1], 'width': im_width, 'height': im_height}
            simpleScene = SimpleScene(cam_dict, bdb3d_corners)
            edges_front_list, face_edges_list, face_verts_list = simpleScene.get_edges_front(bdb3d_corners, ax_3d=None, if_vis=False)
            mask_combined, mask_list, mask_conflict = simpleScene.poly_to_masks(face_verts_list)

            ax_3d = simpleScene.vis_3d(bdb3d_corners)
            ax_2d = simpleScene.vis_2d_bbox_proj(bdb3d_corners, edge_list=[x[0] for x in edges_front_list], if_show=True)
            simpleScene.vis_mask_combined(mask_combined, ax_2d=ax_2d)

            invd_list = simpleScene.param_planes()
            depth_masked_list = [face_idx_mask[1]/invd_list[face_idx_mask[0]] for face_idx_mask in mask_list]
            depth_combined = np.sum(np.stack(depth_masked_list), 0)

            if if_vis:
                fig = plt.figure(figsize=(15, 4))
                plt.subplot(121)
                plt.imshow(depth_combined, cmap='jet')
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(mask_conflict, cmap='jet')
                plt.colorbar()
                # plt.subplot(143)
                # plt.imshow(invd_list[3], cmap='jet')
                # plt.colorbar()
                # plt.subplot(144)
                # plt.imshow(mask_list[3][1])
                plt.show()

        if not self.if_mute_print:
            print("[draw_projected_depth] Returned.")

        return depth_combined, mask_conflict

    def draw_3D_scene_plt(self, type = 'prediction', if_save = True, save_path='', fig_or_ax=[None, None],  which_to_vis='cell_info', \
            if_show_emitter=True, if_show_objs=True, if_return_cells_vis_info=False, hide_cells=False, hide_random_id=True, scale_emitter_length=1., \
            if_debug=False, if_dump_to_mesh=False, fig_scale=1., pickle_id=0):
        assert type in ['prediction', 'GT', 'both']
        figs_to_draw = {'prediction': ['prediction'], 'GT': ['GT'],'both': ['prediction', 'GT']}
        figs_to_draw = figs_to_draw[type]
        cells_vis_info_list_pred = []
        cells_vis_info_list_GT = []

        ax_3d_GT, ax_3d_PRED = fig_or_ax[0], fig_or_ax[1]

        if_new_fig = ax_3d_GT is None and ax_3d_PRED is None
        if if_new_fig:
            fig = plt.figure(figsize=(15*fig_scale, 8*fig_scale ))

        if 'GT' in figs_to_draw:
            if if_new_fig:
                ax_3d_GT = fig.add_subplot(121, projection='3d')
                # ax_3d_GT = fig.gca(projection='3d')
            ax_3d_GT.set_proj_type('ortho')
            ax_3d_GT.set_aspect("auto")
            ax_3d_GT.view_init(elev=-42, azim=111)
            ax_3d_GT.set_title('GT')

        if 'prediction' in figs_to_draw:
            if if_new_fig:
                ax_3d_PRED = fig.add_subplot(122, projection='3d')
                # ax_3d_PRED = fig.gca(projection='3d')
            ax_3d_PRED.set_proj_type('ortho')
            ax_3d_PRED.set_aspect("auto")
            ax_3d_PRED.view_init(elev=-42, azim=111)
            ax_3d_PRED.set_title('PRED')

        # === draw layout, camera and axis

        if type == 'prediction':
            axes = [ax_3d_PRED]
            boxes = [self.pre_layout]
            cam_Rs = [self.pre_cam_R]
        elif type == 'GT':
            axes = [ax_3d_GT]
            boxes = [self.gt_layout]
            cam_Rs = [self.gt_cam_R]
        elif type == 'both':
            axes = [ax_3d_PRED, ax_3d_GT]
            boxes = [self.pre_layout, self.gt_layout]
            cam_Rs = [self.pre_cam_R, self.gt_cam_R]

        for ax_3d, layout, cam_R in zip(axes, boxes, cam_Rs):
            if ax_3d is None:
                continue
            cam_xaxis, cam_yaxis, cam_zaxis = np.split(cam_R, 3, axis=1)
            cam_up = cam_yaxis
            cam_origin = np.zeros_like(cam_up)
            cam_lookat = cam_origin + cam_xaxis
            vis_axis_xyz(ax_3d, cam_xaxis.flatten(), cam_yaxis.flatten(), cam_zaxis.flatten(), cam_origin.flatten(), suffix='_c')
            a = Arrow3D([cam_origin[0][0], cam_lookat[0][0]*2-cam_origin[0][0]], [cam_origin[1][0], cam_lookat[1][0]*2-cam_origin[1][0]], [cam_origin[2][0], cam_lookat[2][0]*2-cam_origin[2][0]], mutation_scale=20,
                            lw=1, arrowstyle="->", color="b")
            ax_3d.add_artist(a)
            a_up = Arrow3D([cam_origin[0][0], cam_origin[0][0]+cam_up[0][0]], [cam_origin[1][0], cam_origin[1][0]+cam_up[1][0]], [cam_origin[2][0], cam_origin[2][0]+cam_up[2][0]], mutation_scale=20,
                            lw=1, arrowstyle="->", color="r")
            ax_3d.add_artist(a_up)
            vis_axis(ax_3d)

            # === draw layout
            assert layout is not None
            vis_cube_plt(layout, ax_3d, 'k', '--', if_face_idx_text=True, if_vertex_idx_text=True, highlight_faces=[0]) # highlight ceiling (face 0) edges

            # === draw emitters
            if self.emitters_obj_list is not None and if_show_emitter:
                for obj_idx, emitter_dict in enumerate(self.emitters_obj_list):
                    #     cat_id, cat_name, cat_color = emitter_dict['catInt_%s'%OR], emitter_dict['catStr_%s'%OR], emitter_dict['catColor_%s'%OR]
                    # else:
                    cat_id, cat_name, cat_color = emitter_dict['cat_id'], emitter_dict['cat_name'], emitter_dict['cat_color']
                    # cat_id, cat_name, cat_color = 1, 'emitter', [0., 1., 0.]
                    if emitter_dict['emitter_prop']['if_lit_up']:
                        cat_name = cat_name + '***'
                    else:
                        cat_name = cat_name + '*--'
                    linestyle = '-.'

                    if cat_id == 0:
                        continue

                    # print('---', emitter_dict['random_id'])
                    obj_label_show = cat_name if hide_random_id else cat_name+'-'+ emitter_dict['random_id']
                    vis_cube_plt(emitter_dict['obj_box_3d'], ax_3d, cat_color, linestyle, obj_label_show)

                    vis_emitter_part = True
                    if emitter_dict['emitter_prop']['if_lit_up'] and emitter_dict['emitter_prop']['obj_type'] == 'window':
                        intensity = emitter_dict['emitter_prop']['emitter_rgb_float']
                        scale = max(intensity) / 255.
                        intensity_scaled = [np.clip(x / scale / 255., 0., 1.) for x in intensity]
                        intensity_scalelog = np.array(intensity).flatten()
                        intensity_scalelog = np.log(np.clip(np.linalg.norm(intensity_scalelog) + 1., 1., np.inf)) / 3. + 0.5 # add 0.5 for vis (otherwise could be too short)
                        # print(intensity, scale, intensity_scaled)

                        if 'light_world_total3d_centeraxis' in emitter_dict: # in total3d generated emitter pickles
                            light_center = emitter_dict['light_world_total3d_centeraxis'][0].flatten()
                            if vis_emitter_part:
                                light_center = np.mean(emitter_dict['bdb3D_emitter_part'], 0).flatten()
                            light_axis = emitter_dict['light_world_total3d_centeraxis'][1].flatten()
                        else: # in RAW frame_dict
                            light_center = emitter_dict['emitter_prop']['light_center_world'].flatten()
                            if vis_emitter_part:
                                light_center = np.mean(emitter_dict['bdb3D_emitter_part'], 0).flatten()
                            light_axis = emitter_dict['emitter_prop']['light_axis_world'].flatten()
                        # intensity_scalelog = 5. 
                        light_axis_length_vis = intensity_scalelog + 2.
                        light_axis_end = light_axis / np.linalg.norm(light_axis) * light_axis_length_vis * scale_emitter_length + light_center
                        # light_axis_end = light_axis / np.linalg.norm(light_axis) * 5 + light_center
                        a_light = Arrow3D([light_center[0], light_axis_end[0]], [light_center[1], light_axis_end[1]], [light_center[2], light_axis_end[2]], mutation_scale=20,
                            lw=2, arrowstyle="-|>", facecolor=intensity_scaled, edgecolor='k')
                        ax_3d.add_artist(a_light)

        if fig_or_ax != [None, None]:
            for ax_3d in [ax_3d_GT, ax_3d_PRED]:
                if ax_3d is not None:
                    ax_3d.set_box_aspect([1,1,1])
                    set_axes_equal(ax_3d) # IMPORTANT - this is also required
            return

        # === draw objs
        cell_info_grid_dict = {'GT': self.cell_info_grid_GT, 'prediction': self.cell_info_grid_PRED}
        if type == 'prediction':
            layout_list = [self.pre_layout]
            boxes_list = [self.pre_boxes]
            cam_Rs = [self.pre_cam_R]
            colors = [[1., 0., 0.]]
            types = ['prediction']
            line_widths = [5]
            linestyles = ['-']
            fontsizes = [15]
            ax_3ds = [ax_3d_PRED]
            cells_vis_info_lists = [cells_vis_info_list_pred]
        elif type == 'both':
            layout_list = [self.pre_layout, self.gt_layout]
            boxes_list = [self.pre_boxes, self.gt_boxes]
            cam_Rs = [self.pre_cam_R, self.gt_cam_R]    
            colors = [[1., 0., 0.], [0., 0., 1.]]
            types = ['prediction', 'GT']
            line_widths = [5, 3]
            linestyles = ['-', '--']
            fontsizes = [15, 12]
            ax_3ds = [ax_3d_PRED, ax_3d_GT]
            cells_vis_info_lists = [cells_vis_info_list_pred, cells_vis_info_list_GT]
        elif type == 'GT':
            layout_list = [self.gt_layout]
            boxes_list = [self.gt_boxes]
            cam_Rs = [self.gt_cam_R]
            colors = [[0., 0., 1.]]
            types = ['GT']
            line_widths = [3]
            linestyles = ['--']
            fontsizes = [12]
            ax_3ds = [ax_3d_GT]
            cells_vis_info_lists = [cells_vis_info_list_GT]
        else:
            assert False, 'not valid Type!'

        for boxes, cam_R, line_width, linestyle, fontsize, current_type, ax_3d in zip(boxes_list, cam_Rs, line_widths, linestyles, fontsizes, types, ax_3ds):
            if boxes is None or not(if_show_objs):
                continue
            
            valid_bbox_idxes = []
            # for bbox_idx, (coeffs, centroid, class_id, basis, random_id) in enumerate(zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis'], boxes['random_id'])):
            for bbox_idx, (coeffs, centroid, class_id, basis) in enumerate(zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis'])):
                # if class_id != 21: 
                #     continue

                # if random_id != 'XRZ7U':
                #     continue

                if_box_valid = self.gt_boxes['if_valid'][bbox_idx]
                if_box_invalid_cat = class_id not in self.valid_class_ids

                message_strs = []
                if if_box_valid==False:
                    message_strs.append('[invalid obj]')
                if if_box_invalid_cat:
                    message_strs.append('[invalid class of dataset %s]'%self.dataset)
                message_str = '-'.join(message_strs)
                message_str = '%d:%s %s'%(class_id, self.classes[class_id], message_str)

                if if_box_invalid_cat or (not if_box_valid):
                    if self.if_hide_invalid_cats:
                        print(magenta('%s [Skipped] %s'%(self.description, message_str)))
                        continue
                    else:
                        print(yellow('%s [Warning] %s'%(self.description, message_str)))
                        linestyle = 'dashdot'
                        color = 'grey'

                bdb3d_corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)

                color = [x/255. for x in self.color_palette[class_id]]
                vis_cube_plt(bdb3d_corners, ax_3d, color, linestyle, self.classes[class_id])
                # print('Showing obj', self.classes[class_id])
                
                for axis_idx, color in zip([0, 1, 2], ['r', 'g', 'b']):
                    a_x = Arrow3D([centroid[0], centroid[0]+basis[axis_idx][0]], [centroid[1], centroid[1]+basis[axis_idx][1]], [centroid[2], centroid[2]+basis[axis_idx][2]], mutation_scale=1,
                            lw=1, arrowstyle="Simple", color=color)
                    ax_3d.add_artist(a_x)

                valid_bbox_idxes.append(bbox_idx)

            if if_dump_to_mesh:
                obj_path_normalized_paths = self.gt_meshes if current_type=='GT' else [x[0] for x in self.pre_meshes]
                obj_path_normalized_paths = [obj_path_normalized_paths[x] for x in valid_bbox_idxes]
                # boxes_valid = [[boxes[key][valid_bbox_idx] for key in ['coeffs', 'centroid', 'class_id', 'basis']] for valid_bbox_idx in valid_bbox_idxes]


                boxes_valid = {}
                for key in ['coeffs', 'centroid', 'class_id', 'basis', 'random_id', 'cat_name']:
                    boxes_valid[key] = [boxes[key][x] for x in valid_bbox_idxes]
                if if_debug:
                    for box_idx, obj_path_normalized_path in enumerate(obj_path_normalized_paths):
                        print(box_idx, boxes_valid['random_id'][box_idx], boxes_valid['cat_name'][box_idx], obj_path_normalized_path)

                # print('=========>')
                # for idx, random_id in enumerate(boxes_valid['random_id']):
                #     if random_id in ['8EFW1', 'NLIQC']:
                #         idx = boxes_valid['random_id'].index(random_id)
                #         print(boxes_valid['cat_name'][idx], random_id)
                #         for key in ['centroid', 'basis', 'coeffs']:
                #             print(key)
                #             print(boxes_valid[key][idx])

                # assert len(obj_path_normalized_paths)==len(boxes_valid)
                # vtk_objects, pre_boxes = format_mesh(obj_path_normalized_paths, boxes_valid)
                [vertices_list, faces_list], bboxes_ = format_mesh(obj_path_normalized_paths, boxes_valid, if_use_vtk=False)
                if len(vertices_list) > 0:
                    vertices_combine = np.vstack(vertices_list)
                    faces_combine = np.vstack(faces_list)
                    if if_debug:
                        scene_mesh_debug_path = 'scene_mesh_debug_val-%d.obj'%self.pickle_id
                    else:
                        scene_mesh_debug_path = Path(self.opt.summary_vis_path_task) / ('scene_mesh-%s-%d.obj'%(current_type, pickle_id))
                    writeMesh(str(scene_mesh_debug_path), vertices_combine, faces_combine)
                    print(white_blue('[%s] Mesh written to '%current_type+str(scene_mesh_debug_path)))
                else:
                    print(yellow('Mesh not written for pickle_id %d: no valid objects'%pickle_id))


        # if not if_show_emitter:
        #     return None, None

        # === draw emitter patches
        # if_vis_lightnet_cells = lightnet_array_GT is not None
        # if if_vis_lightnet_cells:
        #     assert lightnet_array_GT.shape == (6, self.grid_size, self.grid_size, 3)
        if self.emitter2wall_assign_info_list is not None and not hide_cells:
            # basis_indexes = [(1, 0, 2, 3), (4, 5, 7, 6), (0, 1, 4, 5), (1, 5, 2, 6), (3, 2, 7, 6), (4, 0, 7, 3)]
            # constant_axes = [1, 1, 2, 0, 2, 0]
            # basis_v_indexes = [(3, 2, 0), (7, 6, 4), (4, 5, 0), (6, 2, 5), (7, 6, 3), (7, 3, 4)]
            basis_v_indexes = [(3, 2, 0), (7, 4, 6), (4, 0, 5), (6, 5, 2), (7, 6, 3), (7, 3, 4)]


            # face_belong_idx_list = [x['face_belong_idx'] for x in self.emitter2wall_assign_info_list]

            for color, type0, layout, cells_vis_info_list in zip(colors, types, layout_list, cells_vis_info_lists):
                if type0 == 'GT':
                    assert self.emitter_cls_prob_GT is not None
                    if self.emitter_cls_prob_GT is not None:
                        emitter_cls_prob = self.emitter_cls_prob_GT
                else:
                    emitter_cls_prob = self.emitter_cls_prob_PRED
                    emitter_cls_prob = np.clip(emitter_cls_prob, 0., 1.)
                
                assert which_to_vis in ['cell_info', 'cell_prob'], 'Illegal which_to_vis: '+which_to_vis
                cell_info_grid = cell_info_grid_dict[type0]
                if cell_info_grid is None: continue
                assert which_to_vis == 'cell_info', 'others not supported for now!'

                for cell_info in cell_info_grid:
                    wall_idx, i, j = cell_info['wallidx_i_j']

                    origin_v1_v2 = basis_v_indexes[wall_idx]
                    basis_1 = (layout[origin_v1_v2[1]] - layout[origin_v1_v2[0]]) / self.grid_size
                    basis_2 = (layout[origin_v1_v2[2]] - layout[origin_v1_v2[0]]) / self.grid_size
                    origin_0 = layout[origin_v1_v2[0]]
                    
                    x_ij = basis_1 * i + basis_2 * j + origin_0
                    x_i1j = basis_1 * (i+1) + basis_2 * j + origin_0
                    x_i1j1 = basis_1 * (i+1) + basis_2 * (j+1) + origin_0
                    x_ij1 = basis_1 * i + basis_2 * (j+1) + origin_0
                    verts = [[list(x_ij), list(x_i1j), list(x_i1j1), list(x_ij1)]]

                    if which_to_vis == 'cell_info' and cell_info['obj_type'] is None:
                        continue

                    intensity = cell_info['emitter_info']['intensity']
                    intensity_color = [np.clip(x/(max(intensity)+1e-5), 0., 1.) for x in intensity]

                    if which_to_vis == 'cell_info':
                        if cell_info['obj_type'] == 'window':
                            color = 'g'
                            # color = intensity
                        elif cell_info['obj_type'] == 'obj':
                            color = 'b'
                            # color = intensity
                        elif cell_info['obj_type'] == 'null':
                            color = 'm'
                        else:
                            raise ValueError('Invalid: cell_info-obj_type: ' + cell_info['obj_type'])

                        cell_vis = {'alpha': cell_info['light_ratio'], 'color': color, 'idxes': (wall_idx, i, j), 'intensity_color': intensity_color, 'extra_info': cell_info}
                        cell_vis['extra_info'].update({'cell_center': np.mean(np.array(verts).squeeze(), 0).reshape((3, 1)), 'verts': verts})
                        # print(cell_info['light_ratio'], cell_info['obj_type'])
                        # print(verts, np.array(verts).shape)
                        # [0].shape, cell_vis['extra_info']['cell_center'].shape)
                        if cell_info['obj_type'] != 'null':
                            verts = (np.array(verts).squeeze().T - cell_vis['extra_info']['cell_center']) * (cell_vis['alpha']/2.+0.5) + cell_vis['extra_info']['cell_center']
                            verts = [(verts.T).tolist()]
                            poly = Poly3DCollection(verts, facecolor=intensity_color, edgecolor=color)

                            if if_print_log:
                                if type0 == 'GT' and cell_info['obj_type'] == 'window':
                                    # ic('------')
                                    print(wall_idx, i, j)

                        else:
                            poly = Poly3DCollection(verts, facecolor=color)

                    cell_vis.update({'poly': poly})

                    if if_return_cells_vis_info:
                        cells_vis_info_list.append(cell_vis)

                    # draw emitter polys, directions
                    alpha = np.clip(cell_vis['alpha'], 0., 1.)
                    if alpha > 1e-4:
                        # alpha = alpha / 2. + 0.5
                        alpha = 1.
                    if color == 'm':
                        # alpha = alpha / 4.
                        alpha = 0.

                    if_draw_cell = alpha != 0

                    if if_draw_cell:
                        cell_vis['poly'].set_alpha(alpha / 1.2)
                        if type0 == 'GT':
                            ax_3d_GT.add_collection3d(cell_vis['poly'])
                        else:
                            ax_3d_PRED.add_collection3d(cell_vis['poly'])

                        if cell_vis['extra_info'] is not None:
                            extra_info = cell_vis['extra_info']
                            if extra_info and extra_info['obj_type'] == 'window':
                                if 'light_dir_abs' in extra_info['emitter_info']:
                                    light_dir_abs = extra_info['emitter_info']['light_dir_abs']
                                else:
                                    light_dir_offset, normal_outside = extra_info['emitter_info']['light_dir_offset'], extra_info['emitter_info']['normal_outside']
                                    light_dir_abs = light_dir_offset + normal_outside
                                cell_center = extra_info['cell_center'].flatten()
                                if 'intensity_scalelog' in extra_info['emitter_info']:
                                    intensity_scalelog = extra_info['emitter_info']['intensity_scalelog'] / 3. + 0.5 # add 0.5 for vis (otherwise could be too short)
                                else:
                                    # print('2')
                                    # print(extra_info['emitter_info'].keys())
                                    intensity = extra_info['emitter_info']['intensity_scale255'] * np.array(extra_info['emitter_info']['intensity_scaled01']) * 255.
                                    intensity_scalelog = np.log(np.clip(np.linalg.norm(intensity.flatten()) + 1., 1., np.inf)) / 3. + 0.5 # add 0.5 for vis (otherwise could be too short)

                                cell_dir_length = intensity_scalelog + 2.
                                light_end = cell_center + light_dir_abs / np.linalg.norm(light_dir_abs) * cell_dir_length
                                # light_end = cell_center + normal_outside
                                # print(cell_center, light_dir)
                                # print(extra_info['emitter_info'])
                                a = Arrow3D([cell_center[0], light_end[0]], [cell_center[1], light_end[1]], [cell_center[2], light_end[2]], mutation_scale=20,
                                    lw=1, arrowstyle="-|>", facecolor=extra_info['emitter_info']['intensity_scaled01'], edgecolor='grey')
                                if type0 == 'GT':
                                    ax_3d_GT.add_artist(a)
                                else:
                                    ax_3d_PRED.add_artist(a)

    

        for ax_3d in [ax_3d_GT, ax_3d_PRED]:
            if ax_3d is not None:
                ax_3d.set_box_aspect([1,1,1])
                set_axes_equal(ax_3d) # IMPORTANT - this is also required

        if fig_or_ax == [None, None]    :
            return fig, [ax_3d_GT, ax_3d_PRED, [cells_vis_info_list_pred, cells_vis_info_list_GT]]
        else:
            return ax_3d, [ax_3d_GT, ax_3d_PRED, [cells_vis_info_list_pred, cells_vis_info_list_GT]]

    def draw_all_cells(self, ax_3d, layout, lightnet_array_GT, alpha=0.5, if_print_log=False, highlight_cells=[]):
        assert lightnet_array_GT.shape == (6, self.grid_size, self.grid_size, 3)
        # basis_v_indexes = [(3, 2, 0), (7, 6, 4), (4, 5, 0), (6, 2, 5), (7, 6, 3), (7, 3, 4)]
        basis_v_indexes = [(3, 2, 0), (7, 4, 6), (4, 0, 5), (6, 5, 2), (7, 6, 3), (7, 3, 4)]


        for wall_idx in range(6):
            for i in range(self.grid_size):                    
                for j in range(self.grid_size):                    
                    origin_v1_v2 = basis_v_indexes[wall_idx]
                    basis_1 = (layout[origin_v1_v2[1]] - layout[origin_v1_v2[0]]) / self.grid_size
                    basis_2 = (layout[origin_v1_v2[2]] - layout[origin_v1_v2[0]]) / self.grid_size
                    origin_0 = layout[origin_v1_v2[0]]
                    
                    x_ij = basis_1 * i + basis_2 * j + origin_0
                    x_i1j = basis_1 * (i+1) + basis_2 * j + origin_0
                    x_i1j1 = basis_1 * (i+1) + basis_2 * (j+1) + origin_0
                    x_ij1 = basis_1 * i + basis_2 * (j+1) + origin_0
                    verts = [[list(x_ij), list(x_i1j), list(x_i1j1), list(x_ij1)]]

                    verts = np.array(verts).squeeze()
                    verts = [verts.tolist()]
                    if_highlight = (wall_idx, i, j) in highlight_cells
                    poly = Poly3DCollection(verts, facecolor=lightnet_array_GT[wall_idx, i, j, :].flatten().tolist(), edgecolor='r' if if_highlight else 'k', linewidths=3 if if_highlight else 1)

                    cell_vis = {}
                    cell_vis.update({'poly': poly})
                    cell_vis['poly'].set_alpha(alpha)
                    ax_3d.add_collection3d(cell_vis['poly'])


    def draw_projected_layout(self, type = 'prediction', return_plt = False, if_save = True, save_path='', if_use_plt=False, fig_or_ax=None, cam_K_override=None, if_original_lim=False, override_img=None):
        if cam_K_override is not None:
            cam_K = cam_K_override
        else:
            cam_K = self.cam_K
            
        if override_img is None:
            im_uint8 = self.img_map[:]
        else:
            im_uint8 = override_img

        if if_use_plt:
            if fig_or_ax is None:
                fig_2d = plt.figure(figsize=(15, 8))
                ax_2d = fig_2d.gca()
            else:
                ax_2d = fig_or_ax
            im_height, im_width = im_uint8.shape[:2]
            ax_2d.imshow(im_uint8)
            if not if_original_lim:
                ax_2d.set_xlim([-im_width*0.5, im_width*1.5])
                ax_2d.set_ylim([im_height*1.5, -im_height*0.5])
            else:
                ax_2d.set_xlim([0, im_width])
                ax_2d.set_ylim([im_height, 0])
        else:
            img_map = Image.fromarray(im_uint8)
            draw = ImageDraw.Draw(img_map)

        if type == 'prediction':
            boxes = [self.pre_layout]
            cam_Rs = [self.pre_cam_R]
            colors = [[1., 0., 0.]]
            line_widths = [5]
            linestyles = ['-']
        elif type == 'both':
            boxes = [self.pre_layout, self.gt_layout]
            cam_Rs = [self.pre_cam_R, self.gt_cam_R]
            colors = [[1., 0., 0.], [0., 0., 1.]]
            line_widths = [5, 3]
            linestyles = ['-', '--']
        elif type == 'GT':
            boxes = [self.gt_layout]
            cam_Rs = [self.gt_cam_R]
            colors = [[0., 0., 1.]]
            line_widths = [3]
            linestyles = ['--']
        else:
            assert False, 'not valid Type!'

        # for coeffs, centroid, class_id, basis in zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis']):
        for box, cam_R, color, line_width, linestyle in zip(boxes, cam_Rs, colors, line_widths, linestyles):
            if box is None:
                print('[draw_projected_layout] box is None for vis type: %s; skipped'%type)
                continue
            if not isinstance(box, dict):
                box = format_layout(box)
            coeffs, centroid, class_id, basis  = box['coeffs'], box['centroid'], -1, box['basis']
            # if class_id not in RECON_3D_CLS:
            #     continue
            # print(cam_K)
            # center_from_3D, invalid_ids = proj_from_point_to_2d(centroid, cam_K, cam_R)
            bdb3d_corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            bdb2D_from_3D = proj_from_point_to_2d(bdb3d_corners, cam_K, cam_R)[0]

            # bdb2D_from_3D = np.round(bdb2D_from_3D).astype('int32')
            bdb2D_from_3D = [tuple(item) for item in bdb2D_from_3D]

            # color = nyu_color_palette[class_id]
            # color = [1., 0., 0.]

            if if_use_plt:
                extra_transform_matrix = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
                layout_wireframes_proj_list = []
                for idx, idx_list in enumerate([[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]):
                # for idx_list in [[3, 0]]:
                # for idx_list in [[5,6]]:
                    v3d_array = bdb3d_corners
                    for i in range(len(idx_list)-1):
                        x1x2 = np.vstack((v3d_array[idx_list[i]], v3d_array[idx_list[i+1]]))
                        # print(cam_dict)
                        # cam_K[0][0] = -cam_K[0][0]
                        # cam_K[1][1] = -cam_K[1][1]
                        front_axis = cam_R[:, self.cam_fromt_axis_id:self.cam_fromt_axis_id+1]

                        x1x2_proj = project_3d_line(x1x2, cam_R.T, np.zeros((3, 1), dtype=np.float32), cam_K, front_axis*0.01, front_axis, extra_transform_matrix=extra_transform_matrix)
                        layout_wireframes_proj_list.append(x1x2_proj)

                for x1x2_proj in layout_wireframes_proj_list:
                    if x1x2_proj is not None:
                        # color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                        ax_2d.plot([x1x2_proj[0][0], x1x2_proj[1][0]], [x1x2_proj[0][1], x1x2_proj[1][1]], color=color, linewidth=line_width, linestyle=linestyle)

            else:
                draw.line([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[2], bdb2D_from_3D[3], bdb2D_from_3D[0]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                draw.line([bdb2D_from_3D[4], bdb2D_from_3D[5], bdb2D_from_3D[6], bdb2D_from_3D[7], bdb2D_from_3D[4]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                draw.line([bdb2D_from_3D[0], bdb2D_from_3D[4]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                draw.line([bdb2D_from_3D[1], bdb2D_from_3D[5]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                draw.line([bdb2D_from_3D[2], bdb2D_from_3D[6]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                draw.line([bdb2D_from_3D[3], bdb2D_from_3D[7]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)

            # draw.text(tuple(center_from_3D), NYU40CLASSES[class_id],
            #             fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20))

        if return_plt:
            if not self.if_mute_print:
                print("[draw_projected_layout] Returned.")
            if if_use_plt:
                if fig_or_ax is None:
                    return fig_2d, bdb2D_from_3D
                else:
                    return ax_2d, bdb2D_from_3D
            else:
                return img_map, bdb2D_from_3D
        else:
            img_map.show()
            if not self.if_mute_print:
                print("[draw_projected_layout] Shown.")


        if if_save:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            img_map.save(save_path)

    def draw_projected_bdb3d(self, type = 'prediction', return_plt = False, if_save = True, save_path='', if_use_plt=False, fig_or_ax=None, if_vis_3dbboxproj=True, if_vis_2dbbox=False, if_vis_2dbbox_iou=False, if_vis_2dbbox_random_id=False):

        if True:
            if fig_or_ax is None:
                fig_2d = plt.figure(figsize=(15, 8))
                ax_2d = fig_2d.gca()
            else:
                ax_2d = fig_or_ax
            im_uint8 = self.img_map[:]
            im_height, im_width = im_uint8.shape[:2]
            ax_2d.imshow(im_uint8)
            ax_2d.set_xlim([-im_width*0.5, im_width*1.5])
            ax_2d.set_ylim([im_height*1.5, -im_height*0.5])

        else:
            img_map = Image.fromarray(self.img_map[:])
            draw = ImageDraw.Draw(img_map)

        if type == 'prediction':
            current_types = ['prediction']
            boxes_list = [self.pre_boxes]
            cam_Rs = [self.pre_cam_R]
            line_widths = [3]
            linestyles = ['-']
            fontsizes = [15]
        elif type == 'GT':
            current_types = ['GT']
            boxes_list = [self.gt_boxes]
            cam_Rs = [self.gt_cam_R]
            line_widths = [1]
            linestyles = ['--']
            fontsizes = [12]
        elif type == 'both':
            current_types = ['prediction', 'GT']
            boxes_list = [self.pre_boxes, self.gt_boxes]
            cam_Rs = [self.pre_cam_R, self.gt_cam_R]
            line_widths = [3, 1]
            linestyles = ['-', '--']
            fontsizes = [15, 12]

        # for coeffs, centroid, class_id, basis in zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis']):
        for boxes, cam_R, line_width, linestyle, fontsize, current_type in zip(boxes_list, cam_Rs, line_widths, linestyles, fontsizes, current_types):
            if boxes is None:
                print('[draw_projected_bdb3d] boxes is None for vis type: %s; skipped'%current_type)
                continue

            if if_vis_2dbbox and current_type=='GT':
                assert len(boxes['bdb2d']) == len(boxes['coeffs'])

            for bbox_idx, (coeffs, centroid, class_id, basis, random_id) in enumerate(zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis'], boxes['random_id'])):

                if_box_valid = self.gt_boxes['if_valid'][bbox_idx]
                if_box_invalid_cat = class_id not in self.valid_class_ids
                message_strs = []
                if if_box_valid==False:
                    message_strs.append('[invalid obj]')
                if if_box_invalid_cat:
                    message_strs.append('[invalid class of dataset %s]'%self.dataset)
                message_str = '-'.join(message_strs)
                message_str = '%d:%s %s'%(class_id, self.classes[class_id], message_str)

                if if_box_invalid_cat or (not if_box_valid):
                    if self.if_hide_invalid_cats:
                        print(magenta('%s [Skipped] %s'%(self.description, message_str)))
                        continue
                    else:
                        print(yellow('%s [Warning] %s'%(self.description, message_str)))
                        linestyle = 'dashdot'
                        color = 'grey'

                    # if current_type=='prediction':
                    #     continue
                
                # print(centroid.shape, self.cam_K.shape, cam_R.shape)
                center_from_3D, invalid_ids = proj_from_point_to_2d(centroid, self.cam_K, cam_R)
                bdb3d_corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
                bdb2D_from_3D = proj_from_point_to_2d(bdb3d_corners, self.cam_K, cam_R)[0]

                # bdb2D_from_3D = np.round(bdb2D_from_3D).astype('int32')
                bdb2D_from_3D = [tuple(item) for item in bdb2D_from_3D]

                color = [x/255. for x in self.color_palette[class_id]]


                # print(type, bdb2D_from_3D, color)
                if if_use_plt:
                    if if_vis_3dbboxproj:
                        extra_transform_matrix = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
                        layout_wireframes_proj_list = []
                        for idx, idx_list in enumerate([[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]):
                        # for idx_list in [[3, 0]]:
                        # for idx_list in [[5,6]]:
                            v3d_array = bdb3d_corners
                            for i in range(len(idx_list)-1):
                                x1x2 = np.vstack((v3d_array[idx_list[i]], v3d_array[idx_list[i+1]]))
                                # print(cam_dict)
                                # self.cam_K[0][0] = -self.cam_K[0][0]
                                # self.cam_K[1][1] = -self.cam_K[1][1]
                                front_axis = cam_R[:, self.cam_fromt_axis_id:self.cam_fromt_axis_id+1]

                                x1x2_proj = project_3d_line(x1x2, cam_R.T, np.zeros((3, 1), dtype=np.float32), self.cam_K, front_axis*0.01, front_axis, extra_transform_matrix=extra_transform_matrix)
                                layout_wireframes_proj_list.append(x1x2_proj)

                        for x1x2_proj in layout_wireframes_proj_list:
                            if x1x2_proj is not None:
                                # color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                                ax_2d.plot([x1x2_proj[0][0], x1x2_proj[1][0]], [x1x2_proj[0][1], x1x2_proj[1][1]], color=color, linewidth=line_width, linestyle=linestyle)
                            else:
                                print('x1x2_proj in layout_wireframes_proj_list is None (possibly out of the frame)')

                        ax_2d.text(center_from_3D[0], center_from_3D[1], self.classes[class_id]+('' if not if_vis_2dbbox_random_id else '-'+random_id), color=color, fontsize=fontsize)

                    if if_vis_2dbbox and current_type=='GT':
                        bdb2d = boxes['bdb2d'][bbox_idx]
                        bdb2d = {'x1': bdb2d[0], 'y1': bdb2d[1], 'x2': bdb2d[2], 'y2': bdb2d[3]}
                        rect = patches.Rectangle((bdb2d['x1'], bdb2d['y1']), bdb2d['x2']-bdb2d['x1'], bdb2d['y2']-bdb2d['y1'], linewidth=1, edgecolor=color, facecolor='none', linestyle='-.')
                        ax_2d.add_patch(rect)
                        image_bbox = [0., 0., im_width-1, im_height-1]
                        obj_bbox = [bdb2d['x1'], bdb2d['y1'], bdb2d['x2'], bdb2d['y2']]
                        if if_vis_2dbbox_iou:
                            iou, (interArea, boxAArea, boxBArea) = bb_intersection_over_union(image_bbox, obj_bbox, if_return_areas=True)
                            vis_ratio = interArea / (boxBArea+1e-5)
                            ax_2d.text(bdb2d['x1'], bdb2d['y1'], '%s-%d-%.2f'%(self.classes[class_id], class_id, vis_ratio)+('' if not if_vis_2dbbox_random_id else '-'+random_id), color=color)


                else:
                    draw.line([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[2], bdb2D_from_3D[3], bdb2D_from_3D[0]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                    draw.line([bdb2D_from_3D[4], bdb2D_from_3D[5], bdb2D_from_3D[6], bdb2D_from_3D[7], bdb2D_from_3D[4]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                    draw.line([bdb2D_from_3D[0], bdb2D_from_3D[4]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                    draw.line([bdb2D_from_3D[1], bdb2D_from_3D[5]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                    draw.line([bdb2D_from_3D[2], bdb2D_from_3D[6]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)
                    draw.line([bdb2D_from_3D[3], bdb2D_from_3D[7]],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), width=line_width)

                    draw.text(tuple(center_from_3D), self.classes[class_id],
                            fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20))

        if return_plt:
            if not self.if_mute_print:
                print("[draw_projected_bdb3d] Returned.")
            if if_use_plt:
                if fig_or_ax is None:
                    return fig_2d
                else:
                    return ax_2d
            else:
                return img_map
        else:
            img_map.show()
            if not self.if_mute_print:
                print("[draw_projected_bdb3d] Shown.")


        # if return_plt:
        #     print("[draw_projected_bdb3d] Returned.")
        #     return img_map
        # else:
        #     img_map.show()
        #     print("[draw_projected_bdb3d] Shown.")


        if if_save:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            img_map.save(save_path)

    def get_bbox_actor(self, box, color, opacity):
        vectors = [box['coeffs'][basis_id] * basis for basis_id, basis in enumerate(box['basis'])]
        corners, faces = self.get_box_corners(box['centroid'], vectors)
        bbox_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
        bbox_actor.GetProperty().SetOpacity(opacity)
        return bbox_actor

    def get_bbox_line_actor(self, box, color, opacity, width=10):
        vectors = [box['coeffs'][basis_id] * basis for basis_id, basis in enumerate(box['basis'])]
        corners, faces = self.get_box_corners(box['centroid'], vectors)
        bbox_actor = self.set_actor(self.set_mapper(self.set_bbox_line_actor(corners, faces, color), 'box'))
        bbox_actor.GetProperty().SetOpacity(opacity)
        bbox_actor.GetProperty().SetLineWidth(width)
        return bbox_actor

    def get_orientation_actor(self, centroid, vector, color):

        arrow_actor = self.set_arrow_actor(centroid, vector)
        arrow_actor.GetProperty().SetColor(color)

        return arrow_actor

    def get_voxel_actor(self, voxels, voxel_vector, color):
        # draw each voxel
        voxel_actors = []
        for point in voxels:
            corners, faces = self.get_box_corners(point, voxel_vector)
            voxel_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
            voxel_actors.append(voxel_actor)
        return voxel_actors

    def set_render(self):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        # '''draw layout system'''
        # renderer.AddActor(self.set_axes_actor())

        '''draw gt camera orientation'''
        if self.mode == 'gt' or self.mode == 'both':
            color = [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]
            center = [0, 0, 0]
            vectors = self.gt_cam_R.T
            # for index in range(vectors.shape[0]):
            #     arrow_actor = self.set_arrow_actor(center, vectors[index])
            #     arrow_actor.GetProperty().SetColor(color[index])
            #     renderer.AddActor(arrow_actor)
            '''set camera property'''
            camera = self.set_camera(center, vectors, self.cam_K)
            renderer.SetActiveCamera(camera)

        '''draw predicted camera orientation'''
        if self.mode == 'prediction' or self.mode == 'both':
            color = [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]
            center = [0, 0, 0]
            vectors = self.pre_cam_R.T
            # for index in range(vectors.shape[0]):
            #     arrow_actor = self.set_arrow_actor(center, vectors[index])
            #     arrow_actor.GetProperty().SetColor(color[index])
            #     renderer.AddActor(arrow_actor)
            '''set camera property'''
            camera = self.set_camera(center, vectors, self.cam_K)
            renderer.SetActiveCamera(camera)

        '''draw gt layout'''
        if self.mode == 'gt' or self.mode == 'both':
            color = (255, 0, 0)
            opacity = 0.2
            layout_actor = self.get_bbox_actor(self.gt_layout, color, opacity)
            renderer.AddActor(layout_actor)
            layout_line_actor = self.get_bbox_line_actor(self.gt_layout, color, 1.)
            renderer.AddActor(layout_line_actor)

        '''draw predicted layout'''
        if self.mode == 'prediction' or self.mode == 'both':
            color = (75, 75, 75)
            opacity = 0.2
            layout_actor = self.get_bbox_actor(self.pre_layout, color, opacity)
            renderer.AddActor(layout_actor)
            layout_line_actor = self.get_bbox_line_actor(self.pre_layout, (75,75,75), 1.)
            renderer.AddActor(layout_line_actor)

        '''draw gt obj bounding boxes'''
        if self.mode == 'gt' or self.mode == 'both':
            for coeffs, centroid, class_id, basis in zip(self.gt_boxes['coeffs'],
                                                         self.gt_boxes['centroid'],
                                                         self.gt_boxes['class_id'],
                                                         self.gt_boxes['basis']):
                if class_id not in self.valid_class_ids:
                    continue
                color = [1., 0., 0.]
                opacity = 0.2
                box = {'coeffs':coeffs, 'centroid':centroid, 'class_id':class_id, 'basis':basis}
                bbox_actor = self.get_bbox_actor(box, color, opacity)
                renderer.AddActor(bbox_actor)

                # draw orientations
                color = [[0.8, 0.8, 0.8],[0.8, 0.8, 0.8],[1., 0., 0.]]
                vectors = [box['coeffs'][v_id] * vector for v_id, vector in enumerate(box['basis'])]

                for index in range(3):
                    arrow_actor = self.get_orientation_actor(box['centroid'], vectors[index], color[index])
                    renderer.AddActor(arrow_actor)

        '''draw predicted obj bounding boxes'''
        if self.mode == 'prediction' or self.mode == 'both':
            for coeffs, centroid, class_id, basis in zip(self.pre_boxes['coeffs'],
                                                         self.pre_boxes['centroid'],
                                                         self.pre_boxes['class_id'],
                                                         self.pre_boxes['basis']):
                if class_id not in self.valid_class_ids:
                    continue
                color = self.color_palette[class_id]
                opacity = 0.2
                box = {'coeffs':coeffs, 'centroid':centroid, 'class_id':class_id, 'basis':basis}
                bbox_actor = self.get_bbox_actor(box, color, opacity)
                renderer.AddActor(bbox_actor)

                # draw orientations
                color = [[0.8, 0.8, 0.8],[0.8, 0.8, 0.8],[1., 0., 0.]]
                vectors = [box['coeffs'][v_id] * vector for v_id, vector in enumerate(box['basis'])]

                for index in range(3):
                    arrow_actor = self.get_orientation_actor(box['centroid'], vectors[index], color[index])
                    renderer.AddActor(arrow_actor)

        # draw mesh
        if self.mode == 'prediction' and self.output_mesh:
            for obj_idx, class_id in enumerate(self.pre_boxes['class_id']):
                if class_id not in self.valid_class_ids:
                    continue
                color = self.color_palette[class_id]

                object = self.output_mesh[obj_idx]

                object_actor = self.set_actor(self.set_mapper(object, 'model'))
                object_actor.GetProperty().SetColor(color)
                renderer.AddActor(object_actor)

        # '''draw point cloud'''
        # point_actor = self.set_actor(self.set_mapper(self.set_points_property(np.eye(3)), 'box'))
        # point_actor.GetProperty().SetPointSize(1)
        # renderer.AddActor(point_actor)

        renderer.SetBackground(1., 1., 1.)

        return renderer, None

    def draw3D(self, if_save, save_path):
        '''
        Visualize 3D models with their bounding boxes.
        '''
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window = self.set_render_window()
        render_window_interactor.SetRenderWindow(render_window)
        render_window.Render()
        render_window_interactor.Start()

        if if_save:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            im = vtk.vtkWindowToImageFilter()
            writer = vtk.vtkPNGWriter()
            im.SetInput(render_window)
            im.Update()
            writer.SetInputConnection(im.GetOutputPort())
            writer.SetFileName(save_path)
            writer.Write()

