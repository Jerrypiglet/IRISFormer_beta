from re import A
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
from copy import deepcopy
import math

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
from utils.utils_total3D.utils_OR_vis_tools import get_bdb_form_from_corners, format_bboxes, format_layout, format_mesh
from utils.utils_total3D.utils_OR_mesh import writeMesh_rect
from utils.utils_total3D.utils_OR_write_shape_to_xml import transformToXml, addShape, addAreaLight, addMaterial_diffuse
from utils.utils_total3D.sunrgbd_utils import get_corners_of_bb3d_no_index
from utils.utils_total3D.utils_OR_xml import get_XML_root

from utils.utils_total3D import SGOptim

class Box(Scene3D):
    '''
    Visualize of the scene in Total3D coords
    '''

    def __init__(self, img_map, cam_K, gt_cam_R, pred_cam_R, gt_layout, pred_layout, gt_boxes, pre_boxes, gt_meshes_paths, pre_meshes_paths, \
                depth_map=None, normal_map=None, cam_K_scale=1., envmap_params={}, 
                transform_R=None, transform_t=None, hdr_scale=1., env_scale=1., \
                opt=None, dataset='OR', if_hide_invalid_cats=True, description='', if_mute_print=False, OR=None, \
                cam_fromt_axis_id=0, emitters_obj_list_gt=None, \
                emitter2wall_assign_info_list_gt=None, emitter_cls_prob_PRED=None, emitter_cls_prob_GT=None, \
                cell_info_grid_GT=None, cell_info_grid_PRED=None, \
                grid_size=4, paths={}, pickle_id=-1, tid=0, \
                gt_boxes_valid_mask_extra=None, pre_boxes_valid_mask_extra=None, \
                if_use_vtk=False, if_off_screen_vtk=True, \
                if_index_faces_with_basis=True, summary_path='.', material_dict=None):
        super(Scene3D, self).__init__()
        self.opt = opt
        if self.opt is not None:
            self.summary_path = self.opt.summary_vis_path_task
            self.envHeight = self.opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.envHeight
            self.envWidth = self.opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.envWidth
        else:
            self.summary_path = summary_path
            self.envHeight = 8
            self.envWidth = 16

        if 'root_path' in paths:
            self.root_path = paths['root_path']
        else:
            self.root_path = None

        self.envOptim = SGOptim.SGEnvOptimSky(weightValue=np.zeros(3,), thetaValue=0., phiValue=0., ambientValue=np.zeros(3,), isCuda=True)

        self.if_use_vtk = if_use_vtk
        # self.mode = data_type
        # assert self.mode in ['prediction', 'GT', 'both']
        self.if_index_faces_with_basis = if_index_faces_with_basis # if indexing faces with combination of basis of layout, insetad of using self.faces_v_indexes
        
        self.faces_v_indexes = [(3, 2, 0), (7, 4, 6), (4, 0, 5), (6, 5, 2), (7, 6, 3), (7, 3, 4)]
        self.faces_basis_indexes = [['x', 'z', '-y'], ['z', 'x', 'y'], ['y', 'z', 'x'], ['x', '-y', '-z'], ['z', 'y', '-x'], ['x', 'y', 'z']] # 0 for ceiling, 1 for floor

        self._cam_K = cam_K
        self._cam_K_scaled = self.scale_cam_K(cam_K, cam_K_scale)
        self._envmap_params = envmap_params
        self.gt_cam_R = gt_cam_R
        self.pred_cam_R = pred_cam_R
        self.grid_size = grid_size
        
        # both layout and object should be dict of basis, coeffs, and centroid
        self.gt_layout = gt_layout
        self.pred_layout = pred_layout
        self.gt_boxes = gt_boxes
        self.pre_boxes = pre_boxes
        self.gt_boxes_valid = None
        self.pre_boxes_valid = None

        # meshes are paths to objs
        self.gt_meshes_paths = gt_meshes_paths
        self.pre_meshes_paths = pre_meshes_paths

        self.gt_boxes_valid_mask_extra = gt_boxes_valid_mask_extra
        self.pre_boxes_valid_mask_extra = pre_boxes_valid_mask_extra

        self.material_dict = material_dict # global material GT/pred for albedo, roughness

        # --- after post-processing
        self.valid_bbox_idxes_dict = {'prediction': [], 'GT': []}
        self.valid_bbox_meshes_dict = {'prediction': {}, 'GT': {}}
        self.scene_meshes_dict = {'prediction': [], 'GT': []}
        self.layout_info_dict = {'prediction': {}, 'GT': {}}
        self.sg_envmap_info_dict = {'prediction': {}, 'GT': {}}
        self.get_layout_info()
        # self.output_mesh_pred = None
        # self.output_mesh_gt = None
        # if self.mode == 'prediction' and output_mesh is None:
        #     self.output_mesh_pred = output_mesh
        
        self._img_map = img_map
        self._depth_map = depth_map
        self._normal_map = normal_map
        self.im_H, self.im_W = self._img_map.shape[:2]
        if self._depth_map is not None:
            assert self._depth_map.shape==(self.im_H, self.im_W)
        if self._normal_map is not None:
            assert self._normal_map.shape==(self.im_H, self.im_W, 3)
        
        self.dataset = dataset

        self.classes = None
        self.valid_class_ids = None
        self.sample_idx = pickle_id
        self.tid = 0

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
        self.emitters_obj_list_gt = emitters_obj_list_gt
        self.emitter2wall_assign_info_list_gt = emitter2wall_assign_info_list_gt
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

        # ------ transform_to_total3d_coords:RAW -> total3D
        self.transform_R = transform_R
        self.transform_t = transform_t

        # ------ envmaps
        self.hdr_scale = hdr_scale
        self.env_scale = env_scale

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

        self.valid_bbox_idxes = None

    def set_cam_K(self, cam_K):
        self.cam_k = cam_K
    
    def scale_cam_K(self, cam_K, cam_K_scale):
        assert cam_K.shape==(3, 3)
        assert cam_K[2][2]==1.
        if cam_K_scale==1.:
            return cam_K
        else:
            cam_K_scaled = np.vstack([cam_K[:2, :] * cam_K_scale, cam_K[2:3, :]])
            return cam_K_scaled

    def get_valid_class_ids(self):
        return [(x, self.classes[x]) for x in self.valid_class_ids]

    def save_img(self, save_path):
        img_map = Image.fromarray(self.img_map[:])
        img_map.save(str(save_path))

    def get_LightNet_transforms(self, cam_R):
        # based on get_grid_centers()
        self.extra_transform_matrix = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]], dtype=np.float32)
        self.extra_transform_matrix_LightNet = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)
        self.post_transform_matrix = self.extra_transform_matrix @ self.extra_transform_matrix_LightNet

        inv_cam_R_transform_pre = cam_R.T
        inv_transform_matrix_post = self.post_transform_matrix.transpose()
        # # print(inv_transform_matrix_post)

        T_LightNet2Total3D_rightmult = inv_transform_matrix_post @ inv_cam_R_transform_pre
        # T_LightNet2Total3D_rightmult = np.linalg.inv(cam_R.transpose() @ self.post_transform_matrix)
        # print(T_LightNet2Total3D_rightmult)
        # print(inv_transform_matrix_post)
        assert T_LightNet2Total3D_rightmult.shape==(3,3)

        return T_LightNet2Total3D_rightmult

        # return inv_cam_R_transform_pre, inv_transform_matrix_post


    def sample_points(self, sample_every=10, ):
        uu, vv = np.meshgrid(np.arange(0, self.im_W, sample_every), np.arange(0, self.im_H, sample_every))
        z = - self._depth_map[vv, uu]
        u0, v0, f_x, f_y = self._cam_K_scaled[0, 2], self._cam_K_scaled[1, 2], self._cam_K_scaled[0][0], self._cam_K_scaled[1][1]
        x = - (uu - u0) / f_x * z
        y = (vv - v0) / f_y * z
        points_LightNet = np.stack([x.squeeze(), y.squeeze(), z.squeeze()], -1) # [h', w', 3]
        return points_LightNet, uu, vv

    def draw_point_cloud(self, ax_3d, plot_type='GT', sample_every=10, point_scale=2.):
        cam_R = {'GT': self.gt_cam_R, 'prediction': self.pred_cam_R}[plot_type]
        assert self._depth_map is not None
        assert self._normal_map is not None
        # print(self._depth_map.shape, self._normal_map.shape)
        # print(self._cam_K)

        points_LightNet, uu, vv = self.sample_points(sample_every=sample_every)

        h, w = points_LightNet.shape[:2]

        T_LightNet2Total3D_rightmult = self.get_LightNet_transforms(cam_R)

        # print(T_LightNet2Total3D_rightmult)
        
        points_vis = points_LightNet.reshape(-1, 3) @ T_LightNet2Total3D_rightmult
        # points_vis = points_LightNet.reshape(-1, 3)
        # print(np.linalg.det(T_LightNet2Total3D_rightmult))

        # print(points_vis.shape)
        colors = self.img_map[vv, uu, :].reshape(-1, 3).astype(np.float32) / 255.
        # valid_points = points_vis[:, 2] < -0.05
        # print(valid_points.shape)
        p = ax_3d.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], s=point_scale, c=colors[:], depthshade=False)

        return points_vis.reshape(h, w, 3), T_LightNet2Total3D_rightmult

    def draw_SG_lighting(self, ax_3d, axis_LightNet_np, weight_np, sample_every=10, plot_type='GT'):
        '''
        axis_LightNet_np.shape, weight_np.shape, points_vis.shape
        (120, 160, 12, 3), (120, 160, 12, 3), (48, 64, 3)
        '''

        cam_R = {'GT': self.gt_cam_R, 'prediction': self.pred_cam_R}[plot_type]
        points_LightNet, _, _ = self.sample_points(sample_every=sample_every)
        h, w = points_LightNet.shape[:2]
        T_LightNet2Total3D_rightmult = self.get_LightNet_transforms(cam_R)
        points_vis = points_LightNet.reshape(-1, 3) @ T_LightNet2Total3D_rightmult
        points_vis = points_vis.reshape(h, w, 3)

        envRow, envCol, SGNum = self._envmap_params['envRow'], self._envmap_params['envCol'], self._envmap_params['SGNum']
        assert axis_LightNet_np.shape==(envRow, envCol, SGNum, 3)
        assert weight_np.shape==(envRow, envCol, SGNum, 3)
        im_2_env_ratios = self.im_H//envRow, self.im_W//envCol

        for i, ii in enumerate(np.arange(0, self.im_H, sample_every)//im_2_env_ratios[0]):
            for j, jj in enumerate(np.arange(0, self.im_W, sample_every)//im_2_env_ratios[1]):

                # if ii != 100 and jj != 50:
                #     continue

                axis_LightNet_show = axis_LightNet_np[ii, jj] # at half res
                axis_show = axis_LightNet_show @ T_LightNet2Total3D_rightmult

                origin_show = points_vis[i, j]
                weight_show = np.linalg.norm(weight_np[ii, jj], axis=1)
                # print(weight_np.shape, weight_np[ii, jj].shape, weight_show.shape)

                for weight_single, axis_single in zip(weight_show, axis_show):
                    weight_single_vis = weight_single / 200.
                    if weight_single_vis < 0.1:
                        continue
                    # print(weight_single_vis)
                    weight_single_vis = np.clip(weight_single_vis, 0., 5.)

                    a = Arrow3D([origin_show[0], origin_show[0]+axis_single[0]*weight_single_vis], 
                                [origin_show[1], origin_show[1]+axis_single[1]*weight_single_vis], 
                                [origin_show[2], origin_show[2]+axis_single[2]*weight_single_vis], \
                                mutation_scale=20, lw=1, arrowstyle="->", color="b")
                    ax_3d.add_artist(a)


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
            boxes = [self.pred_layout]
            cam_Rs = [self.pred_cam_R]
            colors = [[1., 0., 0.]]
            line_widths = [5]
            linestyles = ['-']
        elif type == 'both':
            boxes = [self.pred_layout, self.gt_layout]
            cam_Rs = [self.pred_cam_R, self.gt_cam_R]
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
            depth_masked_list = [wall_idx_mask[1]/invd_list[wall_idx_mask[0]] for wall_idx_mask in mask_list]
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
    
    def get_layout_info(self):
        for current_type in ['prediction', 'GT']:
            if current_type=='prediction':
                layout = self.pred_layout
            else:
                layout = self.gt_layout

            if layout is None:
                continue

            # print(current_type, layout)

            layout_basis_dict = {'x': layout['basis'][0], 'y': layout['basis'][1], 'z': layout['basis'][2], '-x': -layout['basis'][0], '-y': -layout['basis'][1], '-z': -layout['basis'][2]}
            layout_coeffs_dict = {'x': layout['coeffs'][0], 'y': layout['coeffs'][1], 'z': layout['coeffs'][2], '-x': layout['coeffs'][0], '-y': layout['coeffs'][1], '-z': layout['coeffs'][2]}

            self.layout_info_dict[current_type] = {
                'layout_basis_dict' : layout_basis_dict, 
                'layout_coeffs_dict' : layout_coeffs_dict, 
            }

            for wall_idx in range(6):
                axis_1 = self.faces_basis_indexes[wall_idx][0]
                axis_2 = self.faces_basis_indexes[wall_idx][1]
                axis_3 = self.faces_basis_indexes[wall_idx][2]
                basis_1_unit = layout_basis_dict[axis_1]
                basis_2_unit = layout_basis_dict[axis_2]
                basis_3_unit = layout_basis_dict[axis_3]
                normal_outside = -basis_3_unit
                origin_0 = layout['centroid'] - basis_1_unit * layout_coeffs_dict[axis_1] - basis_2_unit * layout_coeffs_dict[axis_2] - basis_3_unit * layout_coeffs_dict[axis_3]
                basis_1 = basis_1_unit * layout_coeffs_dict[axis_1] * 2 / self.grid_size
                basis_2 = basis_2_unit * layout_coeffs_dict[axis_2] * 2 / self.grid_size
                basis_3 = basis_3_unit * layout_coeffs_dict[axis_3] * 2 / self.grid_size

                self.layout_info_dict[current_type][str(wall_idx)] = {
                    'basis_1_unit' : basis_1_unit, 
                    'basis_2_unit' : basis_2_unit, 
                    'basis_3_unit' : basis_3_unit, 
                    'normal_outside' : normal_outside, 
                    'origin_0' : origin_0, 
                    'basis_1' : basis_1, 
                    'basis_2' : basis_2, 
                    'basis_3' : basis_3, 
                }

    def scene_to_xml(self, split_type='prediction', ):
        assert split_type in ['prediction', 'GT']
        main_xml_file = '/home/ruizhu/Documents/Projects/semanticInverse/train/utils/utils_total3D/sample_xml.xml'
        root = get_XML_root(main_xml_file)
        rec_root = deepcopy(root )

        sg_params_list = self.convert_layout_windows_to_mesh(split_type=split_type, rec_root=rec_root)
        self.sg_envmap_info_dict[split_type]['sg_params_list'] = sg_params_list
        if len(sg_params_list) != 0:
            im_envmap_gen = self.convert_SG_params_to_envmap(sg_params_list)
            self.sg_envmap_info_dict[split_type]['im_envmap_gen'] = im_envmap_gen

        self.convert_lamps_to_mesh(split_type=split_type, rec_root=rec_root)
        self.convert_objs_to_mesh(split_type=split_type, rec_root=rec_root)

        target_xml_file = Path(self.summary_path) / ('xml-%s-tid%d-%d.xml'%(split_type, self.tid, self.sample_idx))
        xmlString = transformToXml(rec_root )
        with open(str(target_xml_file), 'w') as xmlOut:
            xmlOut.write(xmlString )
        print('XML written to %s'%str(target_xml_file))
        return target_xml_file


    def convert_objs_to_mesh(self, split_type='prediction', if_dump_to_mesh=True, if_transform_to_RAW=True, rec_root=None):
        assert split_type in ['prediction', 'GT']
        current_type = split_type

        self.postprocess_objs(split_type=split_type)

        # remove added scene_objs_GT
        # scene_objs_shapes = [x for x in rec_root.findall('shape') if 'scene_obj' in x.get('id')]
        # for x in scene_objs_shapes:
        #     print(x)
        #     rec_root.remove(x)

        for mesh_idx, mesh_obj_dict in enumerate(self.valid_bbox_meshes_dict[current_type]['mesh_objs']):
            vertices = mesh_obj_dict['v']
            if if_transform_to_RAW:
                vertices = self.transform_R.T @ (vertices.T - self.transform_t) # transform back to the RAW world
                vertices = vertices.T
            obj_path = Path(self.summary_path) / ('mesh_obj%d-tid%d-%d.obj'%(mesh_idx, self.tid, self.sample_idx))
            writeMesh(str(obj_path), vertices, mesh_obj_dict['f'])
            print('Written objs to %s'%obj_path)
            # print(mesh_obj_dict['obj_path'],'--->', obj_path)
            self.scene_meshes_dict[current_type].append(obj_path)

            if self.material_dict is not None:
                obj_mask = mesh_obj_dict['obj_mask']
                # material_dict = self.material_dict[current_type]
                material_dict = self.material_dict[split_type]                
                albedo, rough = material_dict['albedo'], material_dict['rough']
                x1, y1, x2, y2 = obj_mask['msk_bdb_half'] # [x1, y1, x2, y2]
                msk_half_bool = obj_mask['msk_half'].astype(np.bool_)
                
                albedo_masked = albedo[y1:y2+1, x1:x2+1]
                rough_masked = rough[y1:y2+1, x1:x2+1]
                # normal_masked = normal[y1:y2+1, x1:x2+1]
                assert(msk_half_bool.shape==albedo_masked.shape[:2])
                albedo_masked = albedo_masked[msk_half_bool]
                rough_masked = rough_masked[msk_half_bool]
                # normal_masked = normal_masked[msk_half_bool]

                albedo_median_diffuse = np.median(albedo_masked, axis=0).flatten().tolist()
                rough_median_diffuse = np.median(rough_masked, axis=0).flatten().item()
                # normal_median_diffuse = np.median(normal_masked, axis=0).flatten()
                # normal_median_diffuse = normal_median_diffuse / (np.linalg.norm(normal_median_diffuse)+1e-6)
                # normal_median_diffuse = normal_median_diffuse.tolist()

            if rec_root is not None:
                obj_id = 'obj_%d'%mesh_idx
                if self.material_dict is not None:
                    addMaterial_diffuse(rec_root, obj_id, albedo_rough_list=[[albedo_median_diffuse, rough_median_diffuse]])
                    rec_root = addShape(rec_root, obj_id, str(obj_path), materials=[[0, 'diffuseAlbedo']], scaleValue=1.)
                else:
                    rec_root = addShape(rec_root, obj_id, str(obj_path), None, scaleValue=1.)


    def convert_lamps_to_mesh(self, split_type='prediction', if_dump_to_mesh=True, if_transform_to_RAW=True, rec_root=None):
        assert split_type in ['prediction', 'GT']
        current_type = split_type
        
        if split_type=='prediction':
            cell_info_grid = self.cell_info_grid_PRED
        else:
            cell_info_grid = self.cell_info_grid_GT


        cells_vis_info_list_filtered = [x for x in cell_info_grid if x['obj_type'] == 'obj']

        for cell_idx, cell_info in enumerate(cells_vis_info_list_filtered):
            if cell_info['light_ratio'] < 0.05:
                continue
            intensity = cell_info['emitter_info']['intensity']
            verts = np.array(cell_info['verts']).squeeze()
            if if_transform_to_RAW:
                verts = self.transform_R.T @ (verts.T - self.transform_t) # transform back to the RAW world
                verts = verts.T
            verts = (verts - np.mean(verts, axis=0, keepdims=True)) * cell_info['light_ratio'] + np.mean(verts, axis=0, keepdims=True)
            verts[:, 1] = verts[:, 1] - 0.1

            wall_idx, i, j = cell_info['wallidx_i_j']

            layout_info_dict = self.layout_info_dict[current_type][str(wall_idx)]
            basis_1_unit = layout_info_dict['basis_1_unit']
            basis_2_unit = layout_info_dict['basis_2_unit']
            basis_3_unit = layout_info_dict['basis_3_unit']
            normal_outside = layout_info_dict['normal_outside']
            origin_0 = layout_info_dict['origin_0']
            basis_1 = layout_info_dict['basis_1']
            basis_2 = layout_info_dict['basis_2']
            basis_3 = layout_info_dict['basis_3']
            

            edge_length = np.sqrt(cell_info['light_ratio'])
            box_center = basis_1 * (i+0.5) + basis_2 * (j+0.5) + basis_3 * 0.5 + origin_0
            
            lamp_box = get_corners_of_bb3d_no_index(np.vstack([basis_1_unit, basis_2_unit, basis_3_unit]), [edge_length/2., edge_length/2., edge_length/2.], box_center)
            if if_transform_to_RAW:
                lamp_box = self.transform_R.T @ (lamp_box.T - self.transform_t) # transform back to the RAW world
                lamp_box = lamp_box.T
            
            obj_path = Path(self.summary_path) / ('mesh_lamp_cell%d-tid%d-%d.obj'%(cell_idx, self.tid, self.sample_idx))
            writeMesh_rect(str(obj_path), lamp_box)
            print('Written lamps to %s'%obj_path)
            
            print('[Lamp params] for wall %d:'%wall_idx, intensity)
            if rec_root is not None:
                rec_root = addAreaLight(rec_root, 'lamp_cell_on%d_%d'%(wall_idx, cell_idx), str(obj_path), rgbColor=intensity)


    def get_cell_centers(self, split_type='prediction', if_transform_to_RAW=True):
        assert split_type in ['prediction', 'GT']
        current_type = split_type
        verts_array_list = []

        for wall_idx in range(6):
            layout_info_dict = self.layout_info_dict[current_type][str(wall_idx)]
            origin_0 = layout_info_dict['origin_0']
            basis_1 = layout_info_dict['basis_1']
            basis_2 = layout_info_dict['basis_2']

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x_ij = basis_1 * i + basis_2 * j + origin_0
                    x_i1j = basis_1 * (i+1) + basis_2 * j + origin_0
                    x_i1j1 = basis_1 * (i+1) + basis_2 * (j+1) + origin_0
                    x_ij1 = basis_1 * i + basis_2 * (j+1) + origin_0
                    verts = [[list(x_ij), list(x_i1j), list(x_i1j1), list(x_ij1)]]
                    verts_array = np.array(verts).squeeze() # [4, 3]
                    if if_transform_to_RAW:
                        verts_array = self.transform_R.T @ (verts_array.T - self.transform_t) # transform back to the RAW world
                        verts_array = verts_array.T

                    verts_array_list.append(verts_array)

        return np.stack(verts_array_list)

    def convert_layout_windows_to_mesh(self, split_type='prediction', if_dump_to_mesh=True, if_transform_to_RAW=True, layout_scale=1., rec_root=None):
        assert split_type in ['prediction', 'GT']
        current_type = split_type
        
        if self.scene_meshes_dict[current_type]!=[]:
            return

        if split_type=='prediction':
            # layout = self.pred_layout
            cell_info_grid = self.cell_info_grid_PRED
            # cam_R = self.pred_cam_R
        else:
            # layout = self.gt_layout
            cell_info_grid = self.cell_info_grid_GT
            # cam_R = self.gt_cam_R
    
        if self.material_dict is not None:
            material_dict = self.material_dict[split_type]
            semseg = material_dict['semseg']
            wall_mask = semseg==43
            floor_mask = semseg==44
            ceiling_mask = semseg==45
            albedo, rough = material_dict['albedo'], material_dict['rough']
            layout_mat_dict = {}
            for layout_name, wall_idxes, mask in zip(['ceiling', 'floor', 'wall'], [[0], [1], [2, 3, 4, 5]], [ceiling_mask, floor_mask, wall_mask]):
                albedo_masked = albedo[mask]
                rough_masked = rough[mask]
                if albedo_masked.shape[0]>0:
                    albedo_median_diffuse = np.median(albedo_masked, axis=0).flatten().tolist()
                    print('[!!!!] median albedo for '+layout_name, albedo_median_diffuse)
                else:
                    print('[!!!!] no mask for albedo for '+layout_name)
                    albedo_median_diffuse = [0.5, 0.5, 0.5]
                if rough_masked.shape[0]>0:
                    rough_median_diffuse = np.median(rough_masked, axis=0).flatten().item()
                else:
                    rough_median_diffuse = 0.5
                assert not math.isnan(rough_median_diffuse)

                layout_mat_dict[layout_name] = [wall_idxes, albedo_median_diffuse, rough_median_diffuse]

        sg_params_list = []

        for wall_idx in range(6):
            valid_cell_list = []
            light_ratio_sum = 0.
            
            layout_info_dict = self.layout_info_dict[current_type][str(wall_idx)]
            basis_1_unit = layout_info_dict['basis_1_unit']
            basis_2_unit = layout_info_dict['basis_2_unit']
            basis_3_unit = layout_info_dict['basis_3_unit']
            normal_outside = layout_info_dict['normal_outside']
            origin_0 = layout_info_dict['origin_0']
            basis_1 = layout_info_dict['basis_1']
            basis_2 = layout_info_dict['basis_2']
            basis_3 = layout_info_dict['basis_3']

            for cell_info in cell_info_grid:
                wall_idx_cell, i, j = cell_info['wallidx_i_j']
                if wall_idx_cell != wall_idx:
                    continue

                if not(cell_info['obj_type'] == 'window' and cell_info['light_ratio'] > 0.1):
                    continue

                
                x_ij = basis_1 * i + basis_2 * j + origin_0
                x_i1j = basis_1 * (i+1) + basis_2 * j + origin_0
                x_i1j1 = basis_1 * (i+1) + basis_2 * (j+1) + origin_0
                x_ij1 = basis_1 * i + basis_2 * (j+1) + origin_0
                verts = [[list(x_ij), list(x_i1j), list(x_i1j1), list(x_ij1)]]
                verts_array = np.array(verts).squeeze()
                if if_transform_to_RAW:
                    verts_array = self.transform_R.T @ (verts_array.T - self.transform_t) # transform back to the RAW world
                    verts_array = verts_array.T
                
                light_ratio_sum += cell_info['light_ratio']

                # get SG params
                # light_dir_offset = cell_info['emitter_info']['light_dir_offset']
                # light_dir_abs = light_dir_offset + normal_outside
                if 'light_dir_abs' in cell_info['emitter_info']:
                    light_dir_abs = cell_info['emitter_info']['light_dir_abs']
                else:
                    light_dir_abs = cell_info['emitter_info']['light_dir_offset'] + cell_info['normal_outside']
                light_dir_abs = light_dir_abs / (np.linalg.norm(light_dir_abs)+1e-6)
                if if_transform_to_RAW:
                    light_dir_abs = self.transform_R.T @ light_dir_abs.reshape(3, 1) # transform back to the RAW world
                    light_dir_abs = light_dir_abs.flatten()
                # intensity = cell_info['emitter_info']['intensity'] / self.hdr_scale
                intensity = cell_info['emitter_info']['intensity']
                lamb = cell_info['emitter_info']['lamb']
                if isinstance(lamb, np.ndarray):
                    lamb = lamb.squeeze().item()

                # lamb = 70.

                valid_cell_list.append({'basis_1_unit': basis_1_unit, 'basis_2_unit': basis_2_unit, 'light_ratio': cell_info['light_ratio'], 'center': np.mean(verts_array, axis=0), \
                                        'center_axis1': (verts_array[1]+verts_array[0])/2., 'center_axis2': (verts_array[3]+verts_array[0])/2., 'i': i+0.5, 'j': j+0.5, \
                                        'light_dir_abs': light_dir_abs, 
                                        'intensity': intensity, 
                                        'lamb': lamb, 'wall_idx': wall_idx})
                                        
            if light_ratio_sum < 1.:
                verts_face_total3d = np.stack([origin_0, origin_0+basis_1*self.grid_size, origin_0+basis_1*self.grid_size+basis_2*self.grid_size, origin_0+basis_2*self.grid_size]).reshape((4,3))
                if if_transform_to_RAW:
                    verts_face = self.transform_R.T @ (verts_face_total3d.T - self.transform_t) # transform back to the RAW world
                    verts_face = verts_face.T

                obj_path = Path(self.summary_path) / ('mesh_wall%d-tid%d-%d.obj'%(wall_idx, self.tid, self.sample_idx))
                writeMesh_rect(str(obj_path), verts_face)
                self.scene_meshes_dict[current_type].append(obj_path)
                print('Written wall to %s'%obj_path)
                if rec_root is not None:
                    if self.material_dict is not None:
                        for layout_name in layout_mat_dict:
                            if wall_idx in layout_mat_dict[layout_name][0]:
                                print(wall_idx, '--->', layout_name)
                                albedo_median_diffuse, rough_median_diffuse = layout_mat_dict[layout_name][1], layout_mat_dict[layout_name][2]
                                addMaterial_diffuse(rec_root, 'wall_%d'%wall_idx, albedo_rough_list=[[albedo_median_diffuse, rough_median_diffuse]])
                        rec_root = addShape(rec_root, 'wall_%d'%wall_idx, str(obj_path), materials=[[0, 'diffuseAlbedo']], scaleValue=layout_scale)
                    else:
                        rec_root = addShape(rec_root, 'wall_%d'%wall_idx, str(obj_path), None, scaleValue=layout_scale)
                
            else:
                # merge window cells and write one wall as patches
                axis_1_cell = basis_1
                axis_2_cell = basis_2
                weights = np.array([x['light_ratio'] for x in valid_cell_list])
                weights = weights / np.sum(weights)
                weighted_center = np.sum(weights.reshape(-1, 1) * np.array([x['center'] for x in valid_cell_list]), 0)

                area_ori = np.linalg.norm(axis_1_cell) * np.linalg.norm(axis_2_cell)
                area_sum = sum([x['light_ratio'] for x in valid_cell_list]) * area_ori
                edge_length = np.sqrt(area_sum)

                coeffs_axis1 = (np.array([x['i'] for x in valid_cell_list])).reshape(-1, 1)
                centers_axis1 = np.outer(coeffs_axis1, basis_1.reshape(1, 3))
                weighted_center_axis1 = np.sum(weights.reshape(-1, 1) * centers_axis1, 0) # offset w.r.t. origin_0
                coeffs_axis2 = (np.array([x['j'] for x in valid_cell_list])).reshape(-1, 1)
                centers_axis2 = np.outer(coeffs_axis2, basis_2.reshape(1, 3))
                weighted_center_axis2 = np.sum(weights.reshape(-1, 1) * centers_axis2, 0) # offset w.r.t. origin_0

                weighted_center_axis1_coeff = np.sum(weights.reshape(-1, 1) * coeffs_axis1, 0).item()
                weighted_center_axis2_coeff = np.sum(weights.reshape(-1, 1) * coeffs_axis2, 0).item()
                half_edge_axis1_coeff = edge_length / 2. / np.linalg.norm(basis_1)
                half_edge_axis2_coeff = edge_length / 2. / np.linalg.norm(basis_2)
                c1, c2 = weighted_center_axis1_coeff, weighted_center_axis2_coeff
                h1, h2 = half_edge_axis1_coeff, half_edge_axis2_coeff

                for patch_idx, coeffs in enumerate([\
                                                    [[0., 0.], [c1-h1, 0], [c1-h1, c2-h2], [0., c2-h2]], \
                                                    [[c1-h1, 0], [c1+h1, 0], [c1+h1, c2-h2], [c1-h1, c2-h2]], \
                                                    [[c1+h1, 0], [self.grid_size, 0], [self.grid_size, c2-h2], [c1+h1, c2-h2]], \
                                                    [[0, c2-h2], [c1-h1, c2-h2], [c1-h1, c2+h2], [0, c2+h2]], \
                                                    [[c1+h1, c2-h2], [self.grid_size, c2-h2], [self.grid_size, c2+h2], [c2+h1, c2+h2]], \
                                                    [[0, c2+h2], [c1+h1, c2+h2], [c1+h1, self.grid_size], [0, self.grid_size]], \
                                                    [[c1-h1, c2+h2], [c1+h1, c2+h2], [c1+h1, self.grid_size], [c1-h1, self.grid_size]], \
                                                    [[c1+h1, c2+h2], [self.grid_size, c2+h2], [self.grid_size, self.grid_size], [c1+h1, self.grid_size]]
                                                    ]):
                    patch = np.stack([origin_0 + basis_1*coeff_1 + basis_2*coeff_2 for coeff_1, coeff_2 in coeffs])
                    if if_transform_to_RAW:
                        patch = self.transform_R.T @ (patch.T - self.transform_t) # transform back to the RAW world
                        patch = patch.T
                    obj_path = Path(self.summary_path) / ('mesh_patch%d_on%d-tid%d-%d.obj'%(patch_idx, wall_idx, self.tid, self.sample_idx))

                    writeMesh_rect(str(obj_path), patch)
                    self.scene_meshes_dict[current_type].append(obj_path)
                    print('Written wall to %s'%obj_path)

                    if rec_root is not None:
                        if self.material_dict is not None:
                            for layout_name in layout_mat_dict:
                                if wall_idx in layout_mat_dict[layout_name][0]:
                                    albedo_median_diffuse, rough_median_diffuse = layout_mat_dict[layout_name][1], layout_mat_dict[layout_name][2]
                                    addMaterial_diffuse(rec_root, 'window_patch_on%d_%d'%(wall_idx, patch_idx), albedo_rough_list=[[albedo_median_diffuse, rough_median_diffuse]])
                            rec_root = addShape(rec_root, 'window_patch_on%d_%d'%(wall_idx, patch_idx), str(obj_path), materials=[[0, 'diffuseAlbedo']], scaleValue=layout_scale)
                        else:
                            rec_root = addShape(rec_root, 'window_patch_on%d_%d'%(wall_idx, patch_idx), str(obj_path), None, scaleValue=layout_scale)

            # convert to SG params for envmap
            # for valid_cell in valid_cell_list:
            if len(valid_cell_list) > 0:
                light_dir_abs = np.vstack([x['light_dir_abs'] for x in valid_cell_list]) # [N, 3]
                intensity = np.vstack([x['intensity'] for x in valid_cell_list]) # [N, 3]
                lamb = np.vstack([x['lamb'] for x in valid_cell_list]) # [N.]
                light_dir_abs_median = np.median(light_dir_abs, axis=0)
                intensity_median = np.median(intensity, axis=0)
                lamb_median = np.median(lamb, axis=0)
                print('[SG params] for wall %d:'%wall_idx, light_dir_abs_median, intensity_median, lamb_median)

                sg_params_list.append({'wall_idx': wall_idx, 'intensity': intensity_median, 'lamb': lamb_median, 'light_dir_abs': light_dir_abs_median})
        return sg_params_list

    def convert_SG_params_to_envmap(self, sg_params_list):
        im_envmap_gen_list = []
        for sg_params in sg_params_list:
            # intensity_SG = sg_params['intensity'] / self.hdr_scale / self.env_scale
            intensity_SG = sg_params['intensity']
            lamb_SG = sg_params['lamb']
            light_dir_abs_RAW = sg_params['light_dir_abs']

            weight_SG = intensity_SG
            light_axis_world_SG = np.array([light_dir_abs_RAW[2], -light_dir_abs_RAW[0], light_dir_abs_RAW[1]])
            light_axis_world_SG = light_axis_world_SG / np.linalg.norm(light_axis_world_SG)
            cos_theta = light_axis_world_SG[2]
            theta_SG = np.arccos(cos_theta) # [0, pi]
            cos_phi = light_axis_world_SG[0] / np.sin(theta_SG)
            sin_phi = light_axis_world_SG[1] / np.sin(theta_SG)
            phi_SG = np.arctan2(sin_phi, cos_phi)
            assert phi_SG >= -np.pi and phi_SG <= np.pi
            ambient_SG = np.zeros(3, dtype=np.float32)

            recImg = self.envOptim.renderSG(torch.tensor(theta_SG).reshape((1, 1, 1, 1, 1, 1)).cuda(), torch.tensor(phi_SG).reshape((1, 1, 1, 1, 1, 1)).cuda(), \
                                    torch.tensor(lamb_SG).reshape((1, 1, 1, 1, 1)).cuda(), torch.tensor(weight_SG).reshape((1, 1, 3, 1, 1)).cuda(), \
                                    ambient=torch.tensor(ambient_SG).reshape((1, 1, 3, 1, 1)).cuda())
            im_envmap_gen = recImg.cpu().numpy().squeeze().transpose(1, 2, 0)

            im_envmap_gen_list.append(im_envmap_gen_list)
        
        im_envmap_gen = np.stack([im_envmap_gen, im_envmap_gen]).sum(0)
        return im_envmap_gen

    def postprocess_objs(self, split_type='prediction', if_dump_objs_to_mesh=False, if_dump_scene_to_mesh=False, if_debug=False):
        assert split_type in ['prediction', 'GT', 'both']
        if split_type == 'prediction':
            boxes_list = [self.pre_boxes]
            types = ['prediction']
        elif split_type == 'both':
            boxes_list = [self.pre_boxes, self.gt_boxes]
            types = ['prediction', 'GT']
        elif split_type == 'GT':
            boxes_list = [self.gt_boxes]
            types = ['GT']

        assert not if_dump_objs_to_mesh, 'not implemented yet...'

        for boxes, current_type in zip(boxes_list, types):
            if boxes is None or self.valid_bbox_idxes_dict[current_type]!=[]:
                print('[!!!] Skipped processing boxes for %s'%current_type, boxes is None, self.valid_bbox_idxes_dict[current_type])
                continue

            for bbox_idx, (coeffs, centroid, class_id, basis) in enumerate(zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis'])):
                # assert len(self.gt_boxes['mask'])==len(boxes['coeffs'])
                if_box_valid = self.gt_boxes['if_valid'][bbox_idx]
                if_box_invalid_cat = class_id not in self.valid_class_ids

                message_strs = []
                if not if_box_valid:
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

                self.valid_bbox_idxes_dict[current_type].append(bbox_idx)

            if self.valid_bbox_meshes_dict[current_type]!={}:
                continue

            obj_paths = self.gt_meshes_paths if current_type=='GT' else [x[0] for x in self.pre_meshes_paths]
            if obj_paths is None:
                break
            obj_paths = [obj_paths[x] for x in self.valid_bbox_idxes_dict[current_type]]
            if self.root_path is not None:
                obj_paths = [str(Path(self.root_path) / x) for x in obj_paths]

            boxes_valid = {}
            for key in ['coeffs', 'centroid', 'class_id', 'basis', 'random_id', 'cat_name']:
                boxes_valid[key] = [boxes[key][x] for x in self.valid_bbox_idxes_dict[current_type]]
            if if_debug:
                for box_idx, obj_path_normalized_path in enumerate(obj_paths):
                    print(box_idx, boxes_valid['random_id'][box_idx], boxes_valid['cat_name'][box_idx], obj_path_normalized_path)
            self.valid_bbox_meshes_dict[current_type]['boxes_valid'] = boxes_valid

            [vertices_list, faces_list, faces_notAdd_list], bboxes_ = format_mesh(obj_paths, boxes_valid, if_use_vtk=False)

            if if_dump_scene_to_mesh:
                if len(vertices_list) > 0:
                    vertices_combine = np.vstack(vertices_list)
                    faces_combine = np.vstack(faces_list)
                    if if_debug:
                        scene_mesh_path = 'scene_mesh_debug_val-%d.obj'%self.sample_idx
                    else:
                        scene_mesh_path = Path(self.summary_path) / ('scene_mesh-%s-%d.obj'%(current_type, self.sample_idx))
                    writeMesh(str(scene_mesh_path), vertices_combine, faces_combine)
                    print(white_blue('[%s] Mesh written to '%current_type+str(scene_mesh_path)))
                else:
                    print(yellow('Mesh not written for pickle_id %d: no valid objects'%self.sample_idx))
                self.valid_bbox_meshes_dict[current_type]['mesh_path_scene'] = str(scene_mesh_path)

            self.valid_bbox_meshes_dict[current_type]['mesh_objs'] = []
            for v, f, obj_path, bbox_idx in zip(vertices_list, faces_notAdd_list, obj_paths, self.valid_bbox_idxes_dict[current_type]):
                mesh_obj = {'v': v, 'f': f, 'obj_path': obj_path, 'obj_mask': self.gt_boxes['obj_masks'][bbox_idx]}
                self.valid_bbox_meshes_dict[current_type]['mesh_objs'].append(mesh_obj)
                # if if_dump_objs_to_mesh:
                #     obj_mesh_path = Path(self.opt.summary_vis_path_task) / ('obj_mesh-%s-tid%d-%d.obj'%(current_type, self.tid,     self.sample_idx))

            if self.if_use_vtk: # utils/visualize.py L1194
                vtk_objects, pre_boxes_ = format_mesh(obj_paths, boxes_valid, if_use_vtk=True)
                assert len(obj_paths)==len(vtk_objects.keys())
                self.valid_bbox_meshes_dict[current_type]['vtk_objs'] = vtk_objects

    def draw_3D_scene_plt(self, vis_type = 'prediction', if_save = True, save_path='', fig_or_ax=[None, None],  which_to_vis='cell_info', \
            if_show_emitter=True, if_show_objs=True, if_show_objs_axes=False, if_show_layout_axes=True, if_return_cells_vis_info=False, hide_cells=False, if_show_cell_normals=False, if_show_cell_meshgrid=False, hide_random_id=True, scale_emitter_length=1., \
            if_debug=False, if_dump_to_mesh=False, fig_scale=1., pickle_id=0, \
            points_backproj=None, points_backproj_color=None):
        assert vis_type in ['prediction', 'GT', 'both']
        figs_to_draw = {'prediction': ['prediction'], 'GT': ['GT'],'both': ['prediction', 'GT']}
        figs_to_draw = figs_to_draw[vis_type]
        cells_vis_info_list_pred = []
        cells_vis_info_list_GT = []

        return_dict = {}

        ax_3d_GT, ax_3d_PRED = fig_or_ax[0], fig_or_ax[1]

        if_new_fig = ax_3d_GT is None and ax_3d_PRED is None
        if if_new_fig:
            fig = plt.figure(figsize=(15*fig_scale, 8*fig_scale )) if vis_type=='both' else plt.figure(figsize=(12*fig_scale, 12*fig_scale ))

        if 'GT' in figs_to_draw:
            if if_new_fig:
                ax_3d_GT = fig.add_subplot(121, projection='3d') if vis_type=='both' else fig.add_subplot(111, projection='3d')
                # ax_3d_GT = fig.gca(projection='3d')
            ax_3d_GT.set_proj_type('ortho')
            ax_3d_GT.set_aspect("auto")
            ax_3d_GT.view_init(elev=-42, azim=111)
            ax_3d_GT.set_title('GT')

        if 'prediction' in figs_to_draw:
            if if_new_fig:
                ax_3d_PRED = fig.add_subplot(122, projection='3d') if vis_type=='both' else fig.add_subplot(111, projection='3d')
                # ax_3d_PRED = fig.gca(projection='3d')
            ax_3d_PRED.set_proj_type('ortho')
            ax_3d_PRED.set_aspect("auto")
            ax_3d_PRED.view_init(elev=-42, azim=111)
            ax_3d_PRED.set_title('PRED')

        # === draw layout, camera and axis

        if vis_type == 'prediction':
            axes = [ax_3d_PRED]
            boxes = [self.pred_layout]
            cam_Rs = [self.pred_cam_R]
        elif vis_type == 'GT':
            axes = [ax_3d_GT]
            boxes = [self.gt_layout]
            cam_Rs = [self.gt_cam_R]
        elif vis_type == 'both':
            axes = [ax_3d_PRED, ax_3d_GT]
            boxes = [self.pred_layout, self.gt_layout]
            cam_Rs = [self.pred_cam_R, self.gt_cam_R]

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
            # ic(cam_origin, cam_up)
            ax_3d.add_artist(a_up)
            vis_axis(ax_3d)

            # === draw layout
            assert layout is not None
            vis_cube_plt(layout['bdb3D'], ax_3d, 'k', '--', if_face_idx_text=True, if_vertex_idx_text=True, highlight_faces=[0]) # highlight ceiling (face 0) edges
            if if_show_layout_axes:
                centroid, basis = layout['centroid'], layout['basis']
                for axis_idx, color, axis_name in zip([0, 1, 2], ['r', 'g', 'b'], ['x', 'y', 'z']):
                    a_x = Arrow3D([centroid[0], centroid[0]+basis[axis_idx][0]], [centroid[1], centroid[1]+basis[axis_idx][1]], [centroid[2], centroid[2]+basis[axis_idx][2]], mutation_scale=1,
                            lw=2, arrowstyle="Simple", color=color)
                    ax_3d.add_artist(a_x)
                    ax_3d.text3D(centroid[0]+basis[axis_idx][0], centroid[1]+basis[axis_idx][1], centroid[2]+basis[axis_idx][2], axis_name, color=color, fontsize=10*fig_scale)

            # === draw emitters
            if self.emitters_obj_list_gt is not None and if_show_emitter:
                for obj_idx, emitter_dict in enumerate(self.emitters_obj_list_gt):
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
        if vis_type == 'prediction':
            layout_list = [self.pred_layout]
            boxes_list = [self.pre_boxes]
            cam_Rs = [self.pred_cam_R]
            colors = [[1., 0., 0.]]
            types = ['prediction']
            line_widths = [5]
            linestyles = ['-']
            fontsizes = [15]
            ax_3ds = [ax_3d_PRED]
            cells_vis_info_lists = [cells_vis_info_list_pred]
        elif vis_type == 'both':
            layout_list = [self.pred_layout, self.gt_layout]
            boxes_list = [self.pre_boxes, self.gt_boxes]
            cam_Rs = [self.pred_cam_R, self.gt_cam_R]    
            colors = [[1., 0., 0.], [0., 0., 1.]]
            types = ['prediction', 'GT']
            line_widths = [5, 3]
            linestyles = ['-', '--']
            fontsizes = [15, 12]
            ax_3ds = [ax_3d_PRED, ax_3d_GT]
            cells_vis_info_lists = [cells_vis_info_list_pred, cells_vis_info_list_GT]
        elif vis_type == 'GT':
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
            
            if self.valid_bbox_idxes_dict[current_type] == []:
                self.postprocess_objs(split_type=current_type)
            valid_bbox_idxes = self.valid_bbox_idxes_dict[current_type]

            for bbox_idx, (coeffs, centroid, class_id, basis) in enumerate(zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis'])):
                # if class_id != 21: 
                #     continue

                # if random_id != 'XRZ7U':
                #     continue

                if bbox_idx not in valid_bbox_idxes:
                    continue

                bdb3d_corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)

                color = [x/255. for x in self.color_palette[class_id]]
                vis_cube_plt(bdb3d_corners, ax_3d, color, linestyle, self.classes[class_id])
                # print('Showing obj', self.classes[class_id])
                
                if if_show_objs_axes:
                    for axis_idx, color, axis_name in zip([0, 1, 2], ['r', 'g', 'b'], ['x', 'y', 'z']):
                        a_x = Arrow3D([centroid[0], centroid[0]+basis[axis_idx][0]], [centroid[1], centroid[1]+basis[axis_idx][1]], [centroid[2], centroid[2]+basis[axis_idx][2]], mutation_scale=1,
                                lw=1, arrowstyle="Simple", color=color)
                        ax_3d.add_artist(a_x)

        # if not if_show_emitter:
        #     return None, None

        # === draw emitter patches
        # if_vis_lightnet_cells = lightnet_array_GT is not None
        # if if_vis_lightnet_cells:
        #     assert lightnet_array_GT.shape == (6, self.grid_size, self.grid_size, 3)
        if self.emitter2wall_assign_info_list_gt is not None and not hide_cells:

            # basis_indexes = [(1, 0, 2, 3), (4, 5, 7, 6), (0, 1, 4, 5), (1, 5, 2, 6), (3, 2, 7, 6), (4, 0, 7, 3)]
            # constant_axes = [1, 1, 2, 0, 2, 0]
            # self.faces_v_indexes = [(3, 2, 0), (7, 6, 4), (4, 5, 0), (6, 2, 5), (7, 6, 3), (7, 3, 4)]
            # self.faces_v_indexes = [(3, 2, 0), (7, 4, 6), (4, 0, 5), (6, 5, 2), (7, 6, 3), (7, 3, 4)]


            # face_belong_idx_list = [x['face_belong_idx'] for x in self.emitter2wall_assign_info_list_gt]

            for color, current_type, layout, cells_vis_info_list in zip(colors, types, layout_list, cells_vis_info_lists):
                # layout_basis_dict = {'x': layout['basis'][0], 'y': layout['basis'][1], 'z': layout['basis'][2], '-x': -layout['basis'][0], '-y': -layout['basis'][1], '-z': -layout['basis'][2]}
                # layout_coeffs_dict = {'x': layout['coeffs'][0], 'y': layout['coeffs'][1], 'z': layout['coeffs'][2], '-x': layout['coeffs'][0], '-y': layout['coeffs'][1], '-z': layout['coeffs'][2]}

                if current_type == 'GT':
                    assert self.emitter_cls_prob_GT is not None
                    if self.emitter_cls_prob_GT is not None:
                        emitter_cls_prob = self.emitter_cls_prob_GT
                else:
                    emitter_cls_prob = self.emitter_cls_prob_PRED
                    emitter_cls_prob = np.clip(emitter_cls_prob, 0., 1.)
                
                assert which_to_vis in ['cell_info', 'cell_prob'], 'Illegal which_to_vis: '+which_to_vis
                cell_info_grid = cell_info_grid_dict[current_type]
                if cell_info_grid is None: continue
                assert which_to_vis == 'cell_info', 'others not supported for now!'

                for cell_info in cell_info_grid:
                    wall_idx, i, j = cell_info['wallidx_i_j']

                    if self.if_index_faces_with_basis:
                        layout_info_dict = self.layout_info_dict[current_type][str(wall_idx)]
                        basis_1_unit = layout_info_dict['basis_1_unit']
                        basis_2_unit = layout_info_dict['basis_2_unit']
                        basis_3_unit = layout_info_dict['basis_3_unit']
                        normal_outside = layout_info_dict['normal_outside']
                        origin_0 = layout_info_dict['origin_0']
                        basis_1 = layout_info_dict['basis_1']
                        basis_2 = layout_info_dict['basis_2']
                        basis_3 = layout_info_dict['basis_3']
                    else:
                        origin_v1_v2 = self.faces_v_indexes[wall_idx]
                        basis_1 = (layout[origin_v1_v2[1]] - layout[origin_v1_v2[0]]) / self.grid_size
                        basis_2 = (layout[origin_v1_v2[2]] - layout[origin_v1_v2[0]]) / self.grid_size
                        origin_0 = layout[origin_v1_v2[0]]
                        basis_1_unit = basis_1 / np.linalg.norm(basis_1)
                        basis_2_unit = basis_2 / np.linalg.norm(basis_2)
                        normal_outside = -np.cross(basis_1_unit, basis_2_unit)
                    
                    x_ij = basis_1 * i + basis_2 * j + origin_0
                    x_i1j = basis_1 * (i+1) + basis_2 * j + origin_0
                    x_i1j1 = basis_1 * (i+1) + basis_2 * (j+1) + origin_0
                    x_ij1 = basis_1 * i + basis_2 * (j+1) + origin_0
                    verts = [[list(x_ij), list(x_i1j), list(x_i1j1), list(x_ij1)]]

                    if which_to_vis == 'cell_info' and cell_info['obj_type'] is None:
                        continue

                    intensity = cell_info['emitter_info']['intensity']
                    intensity_color = [np.clip(x/(max(intensity)+1e-5), 0., 1.) for x in intensity]
                    # ic(intensity_color, intensity)

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
                        cell_vis['extra_info'].update({'cell_center': np.mean(np.array(verts).squeeze(), 0).reshape((3, 1)), 'verts': verts, 'normal_outside': normal_outside})
                        # print(cell_info['light_ratio'], cell_info['obj_type'])
                        # print(verts, np.array(verts).shape)
                        # [0].shape, cell_vis['extra_info']['cell_center'].shape)
                        if cell_info['obj_type'] != 'null':
                            verts = (np.array(verts).squeeze().T - cell_vis['extra_info']['cell_center']) * (cell_vis['alpha']/2.+0.5) + cell_vis['extra_info']['cell_center']
                            verts = [(verts.T).tolist()]
                            poly = Poly3DCollection(verts, facecolor=intensity_color, edgecolor=color)

                            if if_debug:
                                if current_type == 'GT' and cell_info['obj_type'] == 'window':
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
                        if current_type == 'GT':
                            ax_3d_GT.add_collection3d(cell_vis['poly'])
                        else:
                            ax_3d_PRED.add_collection3d(cell_vis['poly'])


                        if cell_vis['extra_info'] is not None:
                            extra_info = cell_vis['extra_info']
                            if extra_info and extra_info['obj_type'] == 'window':
                                # normal_outside = extra_info['emitter_info']['normal_outside']
                                # checking meshgrid and normal from LightAccuNet against normal from Layout
                                if 'emitter_outdirs_meshgrid_Total3D_outside_abs' in extra_info['emitter_info']:
                                    emitter_outdirs_meshgrid_Total3D_outside_abs = extra_info['emitter_info']['emitter_outdirs_meshgrid_Total3D_outside_abs']
                                    assert emitter_outdirs_meshgrid_Total3D_outside_abs.shape==(self.envHeight, self.envWidth, 3)
                                    for ii in range(self.envHeight):
                                        for jj in range(self.envWidth):
                                            dot_prod = np.dot(normal_outside.flatten(), emitter_outdirs_meshgrid_Total3D_outside_abs[ii, jj].flatten())
                                            assert dot_prod >= 0
                                if 'normal_outside_Total3D_single' in extra_info['emitter_info']:
                                    assert np.amax(np.abs(extra_info['emitter_info']['normal_outside_Total3D_single'] - normal_outside)) < 1e-3

                                if 'light_dir_abs' in extra_info['emitter_info']:
                                    light_dir_abs = extra_info['emitter_info']['light_dir_abs']
                                else:
                                    light_dir_offset = extra_info['emitter_info']['light_dir_offset']
                                    light_dir_abs = light_dir_offset + normal_outside
                                light_dir_abs = light_dir_abs / (np.linalg.norm(light_dir_abs)+1e-6)

                                cell_center = extra_info['cell_center'].flatten()


                                if 'intensity_scalelog' in extra_info['emitter_info']:
                                    intensity_scalelog = extra_info['emitter_info']['intensity_scalelog'] / 3. + 0.5 # add 1. for vis (otherwise could be too short)
                                else:
                                    # print('2')
                                    # print(extra_info['emitter_info'].keys())
                                    intensity = extra_info['emitter_info']['intensity_scale255'] * np.array(extra_info['emitter_info']['intensity_scaled01']) * 255.
                                    intensity_scalelog = np.log(np.clip(np.linalg.norm(intensity.flatten()) + 1., 1., np.inf)) / 3. + 0.5 # add 1. for vis (otherwise could be too short)

                                cell_dir_length = intensity_scalelog
                                light_end = cell_center + light_dir_abs * cell_dir_length
                                # light_end = cell_center + normal_outside
                                # print(cell_center, light_dir)
                                # print(extra_info['emitter_info'])
                                a = Arrow3D([cell_center[0], light_end[0]], [cell_center[1], light_end[1]], [cell_center[2], light_end[2]], mutation_scale=20,
                                    lw=1, arrowstyle="-|>", facecolor=extra_info['emitter_info']['intensity_scaled01'], edgecolor='grey')
                                if current_type == 'GT':
                                    ax_3d_GT.add_artist(a)
                                else:
                                    ax_3d_PRED.add_artist(a)

                                if if_show_cell_normals and normal_outside is not None: # visualize the normals of cells: https://i.imgur.com/aoJszVa.png
                                    normal_end = cell_center + normal_outside * 10.
                                    a = Arrow3D([cell_center[0], normal_end[0]], [cell_center[1], normal_end[1]], [cell_center[2], normal_end[2]], mutation_scale=20,
                                        lw=1, arrowstyle="->", edgecolor='k')
                                    if current_type == 'GT':
                                        ax_3d_GT.add_artist(a)
                                    else:
                                        ax_3d_PRED.add_artist(a)

                                if if_show_cell_meshgrid and 'emitter_outdirs_meshgrid_Total3D_outside_abs' in extra_info['emitter_info']: # visualize the meshgrid of outer hemisphere of cells: https://i.imgur.com/aoJszVa.png
                                    for ii in range(self.envHeight):
                                        for jj in range(self.envWidth):
                                            meshgrid_end = cell_center + emitter_outdirs_meshgrid_Total3D_outside_abs[ii, jj].flatten()*2
                                            a = Arrow3D([cell_center[0], meshgrid_end[0]], [cell_center[1], meshgrid_end[1]], [cell_center[2], meshgrid_end[2]], mutation_scale=20,
                                                lw=1, arrowstyle="->", edgecolor='k')
                                            if current_type == 'GT':
                                                ax_3d_GT.add_artist(a)
                                            else:
                                                ax_3d_PRED.add_artist(a)

                                            # if normal_outside is not None:
                                            #     dot_prod = np.dot(normal_outside.flatten(), emitter_outdirs_meshgrid_Total3D_outside_abs[ii, jj].flatten())
                                                # assert dot_prod >= 0

        for ax_3d in [ax_3d_GT, ax_3d_PRED]:
            if ax_3d is not None:
                ax_3d.set_box_aspect([1,1,1])
                set_axes_equal(ax_3d) # IMPORTANT - this is also required


        if points_backproj is not None:
            for ii in np.arange(0, points_backproj.shape[0], 10):
                for jj in np.arange(0, points_backproj.shape[1], 10):
                    p = points_backproj[ii, jj]
                    color = (points_backproj_color[ii, jj]).astype(np.float32) / 255.
                    ax_3d_GT.scatter3D(p[0], p[1], p[2], color=color)


        if fig_or_ax == [None, None]    :
            return fig, return_dict, [ax_3d_GT, ax_3d_PRED, [cells_vis_info_list_pred, cells_vis_info_list_GT]]
        else:
            return ax_3d, return_dict, [ax_3d_GT, ax_3d_PRED, [cells_vis_info_list_pred, cells_vis_info_list_GT]]

    def draw_all_cells(self, ax_3d, layout, lightnet_array_GT, current_type='GT', alpha=0.5, if_debug=False, highlight_cells=[]):
        assert lightnet_array_GT.shape == (6, self.grid_size, self.grid_size, 3)
        # self.faces_v_indexes = [(3, 2, 0), (7, 6, 4), (4, 5, 0), (6, 2, 5), (7, 6, 3), (7, 3, 4)]
        # self.faces_v_indexes = [(3, 2, 0), (7, 4, 6), (4, 0, 5), (6, 5, 2), (7, 6, 3), (7, 3, 4)]

        for wall_idx in range(6):
            for i in range(self.grid_size):                    
                for j in range(self.grid_size): 
                    if self.if_index_faces_with_basis:
                        layout_info_dict = self.layout_info_dict[current_type][str(wall_idx)]
                        basis_1_unit = layout_info_dict['basis_1_unit']
                        basis_2_unit = layout_info_dict['basis_2_unit']
                        basis_3_unit = layout_info_dict['basis_3_unit']
                        normal_outside = layout_info_dict['normal_outside']
                        origin_0 = layout_info_dict['origin_0']
                        basis_1 = layout_info_dict['basis_1']
                        basis_2 = layout_info_dict['basis_2']
                        basis_3 = layout_info_dict['basis_3']
                    else:
                        origin_v1_v2 = self.faces_v_indexes[wall_idx]
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
            boxes = [self.pred_layout]
            cam_Rs = [self.pred_cam_R]
            colors = [[1., 0., 0.]]
            line_widths = [5]
            linestyles = ['-']
        elif type == 'both':
            boxes = [self.pred_layout, self.gt_layout]
            cam_Rs = [self.pred_cam_R, self.gt_cam_R]
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
            cam_Rs = [self.pred_cam_R]
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
            cam_Rs = [self.pred_cam_R, self.gt_cam_R]
            line_widths = [3, 1]
            linestyles = ['-', '--']
            fontsizes = [15, 12]

        # for coeffs, centroid, class_id, basis in zip(boxes['coeffs'], boxes['centroid'], boxes['class_id'], boxes['basis']):
        for boxes, cam_R, line_width, linestyle, fontsize, current_type in zip(boxes_list, cam_Rs, line_widths, linestyles, fontsizes, current_types):
            if boxes is None:
                print('[draw_projected_bdb3d] boxes is None for vis type: %s; skipped'%current_type)
                continue
            assert isinstance(boxes, dict)

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
    
    def set_render(self, mode):
        assert mode in ['GT', 'prediction']
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw layout system'''
        # renderer.AddActor(self.set_axes_actor())

        '''draw gt camera orientation'''
        if mode == 'GT':
            cam_R = self.gt_cam_R
        elif mode == 'prediction':
            cam_R = self.pred_cam_R

        color = [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]
        center = [0, 0, 0]
        vectors = cam_R.T
        # print(vectors, mode, self.cam_K)
        # for index in range(vectors.shape[0]):
        #     arrow_actor = self.set_arrow_actor(center, vectors[index])
        #     arrow_actor.GetProperty().SetColor(color[index])
        #     renderer.AddActor(arrow_actor)
        '''set camera property'''
        camera = self.set_camera(center, vectors, self.cam_K)
        renderer.SetActiveCamera(camera)

        # '''draw predicted camera orientation'''
        # if mode == 'prediction' or mode == 'both':
        #     color = [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]
        #     center = [0, 0, 0]
        #     vectors = self.pred_cam_R.T
        #     # for index in range(vectors.shape[0]):
        #     #     arrow_actor = self.set_arrow_actor(center, vectors[index])
        #     #     arrow_actor.GetProperty().SetColor(color[index])
        #     #     renderer.AddActor(arrow_actor)
        #     '''set camera property'''
        #     camera = self.set_camera(center, vectors, self.cam_K)
        #     renderer.SetActiveCamera(camera)

        # '''draw gt layout'''
        # if mode == 'gt' or mode == 'both':
        #     color = (255, 0, 0)
        #     opacity = 0.2
        #     layout_actor = self.get_bbox_actor(self.gt_layout, color, opacity)
        #     renderer.AddActor(layout_actor)
        #     layout_line_actor = self.get_bbox_line_actor(self.gt_layout, color, 1.)
        #     renderer.AddActor(layout_line_actor)

        # '''draw predicted layout'''
        # if mode == 'prediction' or mode == 'both':
        #     color = (75, 75, 75)
        #     opacity = 0.2
        #     layout_actor = self.get_bbox_actor(self.pred_layout, color, opacity)
        #     renderer.AddActor(layout_actor)
        #     layout_line_actor = self.get_bbox_line_actor(self.pred_layout, (75,75,75), 1.)
        #     renderer.AddActor(layout_line_actor)

        # '''draw gt obj bounding boxes'''
        # if mode == 'gt' or mode == 'both':
        #     for coeffs, centroid, class_id, basis in zip(self.gt_boxes['coeffs'],
        #                                                  self.gt_boxes['centroid'],
        #                                                  self.gt_boxes['class_id'],
        #                                                  self.gt_boxes['basis']):
        #         if class_id not in self.valid_class_ids:
        #             continue
        #         color = [1., 0., 0.]
        #         opacity = 0.2
        #         box = {'coeffs':coeffs, 'centroid':centroid, 'class_id':class_id, 'basis':basis}
        #         bbox_actor = self.get_bbox_actor(box, color, opacity)
        #         renderer.AddActor(bbox_actor)

        #         # draw orientations
        #         color = [[0.8, 0.8, 0.8],[0.8, 0.8, 0.8],[1., 0., 0.]]
        #         vectors = [box['coeffs'][v_id] * vector for v_id, vector in enumerate(box['basis'])]

        #         for index in range(3):
        #             arrow_actor = self.get_orientation_actor(box['centroid'], vectors[index], color[index])
        #             renderer.AddActor(arrow_actor)

        # '''draw predicted obj bounding boxes'''
        # if mode == 'prediction' or mode == 'both':
        #     for coeffs, centroid, class_id, basis in zip(self.pre_boxes['coeffs'],
        #                                                  self.pre_boxes['centroid'],
        #                                                  self.pre_boxes['class_id'],
        #                                                  self.pre_boxes['basis']):
        #         if class_id not in self.valid_class_ids:
        #             continue
        #         color = self.color_palette[class_id]
        #         opacity = 0.2
        #         box = {'coeffs':coeffs, 'centroid':centroid, 'class_id':class_id, 'basis':basis}
        #         bbox_actor = self.get_bbox_actor(box, color, opacity)
        #         renderer.AddActor(bbox_actor)

        #         # draw orientations
        #         color = [[0.8, 0.8, 0.8],[0.8, 0.8, 0.8],[1., 0., 0.]]
        #         vectors = [box['coeffs'][v_id] * vector for v_id, vector in enumerate(box['basis'])]

        #         for index in range(3):
        #             arrow_actor = self.get_orientation_actor(box['centroid'], vectors[index], color[index])
        #             renderer.AddActor(arrow_actor)

        # draw mesh
        # if mode == 'prediction':
        #     boxes_valid = self.pre_boxes_valid
        #     output_mesh = self.output_mesh_pred
        # elif mode == 'GT':
        #     boxes_valid = self.gt_boxes_valid
        #     output_mesh = self.output_mesh_gt

        if 'boxes_valid' in self.valid_bbox_meshes_dict[mode]:
            boxes_valid = self.valid_bbox_meshes_dict[mode]['boxes_valid']
            output_mesh = self.valid_bbox_meshes_dict[mode]['vtk_objs']

            for obj_idx, class_id in enumerate(boxes_valid['class_id']):
                # if class_id not in self.valid_class_ids:
                #     continue
                color = self.color_palette[class_id]
                color = (float(color[0])/255., float(color[1])/255., float(color[2])/255.)
                # print(obj_idx, color)

                object = output_mesh[obj_idx]

                object_actor = self.set_actor(self.set_mapper(object, 'model'))
                object_actor.GetProperty().SetColor(color)
                renderer.AddActor(object_actor)
        else:
            print(red('output_mesh is None for mode %s!')%mode)

        # '''draw point cloud'''
        # point_actor = self.set_actor(self.set_mapper(self.set_points_property(np.eye(3)), 'box'))
        # point_actor.GetProperty().SetPointSize(1)
        # renderer.AddActor(point_actor)

        light1 = vtk.vtkLight()
        # light.SetColor(1.0, 1.0, 1.0)
        # light.SetIntensity(1)
        # light.SetPosition(0, 0, 0)
        # light.SetDiffuseColor(1, 1, 1)
        # renderer.AddLight(light)
        light1.SetIntensity(.4)
        light1.SetPosition(5, -5, 5)
        light1.SetDiffuseColor(1, 1, 1)
        light2 = vtk.vtkLight()
        light2.SetIntensity(.2)
        light2.SetPosition(-5, -5, 5)
        light2.SetDiffuseColor(1, 1, 1)
        light3 = vtk.vtkLight()
        light3.SetIntensity(.2)
        light3.SetPosition(5, -5, -5)
        light3.SetDiffuseColor(1, 1, 1)
        light4 = vtk.vtkLight()
        light4.SetIntensity(.4)
        light4.SetPosition(-5, -5, -5)
        light4.SetDiffuseColor(1, 1, 1)
        light5 = vtk.vtkLight()
        light5.SetIntensity(.8)
        light5.SetPosition(0, 5, 0)
        light5.SetDiffuseColor(1, 1, 1)
        light6 = vtk.vtkLight()
        light6.SetIntensity(.8)
        light6.SetPosition(0, 0, 0)
        light6.SetDiffuseColor(1, 1, 1)
        light7 = vtk.vtkLight()
        light7.SetIntensity(.8)
        light7.SetPosition(5, 0, 0)
        light7.SetDiffuseColor(1, 1, 1)
        renderer.AddLight(light1)
        renderer.AddLight(light2)
        renderer.AddLight(light3)
        renderer.AddLight(light4)
        renderer.AddLight(light5)
        renderer.AddLight(light6)
        renderer.AddLight(light7)

        renderer.SetBackground(1., 1., 1.)

        return renderer, None

    def set_render_window(self, mode):
        render_window = vtk.vtkRenderWindow()
        # if self.if_off_screen_vtk:
            # render_window.SetOffScreenRendering(1)
        renderer, voxel_proj = self.set_render(mode=mode)
        render_window.AddRenderer(renderer)
        render_window.SetSize(self.img_map.shape[1], self.img_map.shape[0])

        if isinstance(voxel_proj, np.ndarray):
            plt.imshow(voxel_proj); plt.show()

        return render_window

    def draw3D(self, mode, if_return_img=True, if_save_img=False, save_path_without_suffix='', if_save_obj=False):
        '''
        Visualize 3D models with their bounding boxes.
        '''
        assert mode in ['prediction', 'GT']
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window = self.set_render_window(mode=mode)
        render_window_interactor.SetRenderWindow(render_window)
        render_window.Render()
        render_window_interactor.Start()

        return_dict = {}

        if if_save_img or if_return_img or if_save_obj:
            # if not os.path.exists(os.path.dirname(save_path_without_suffix)):
                # os.makedirs(os.path.dirname(save_path_without_suffix))
            im = vtk.vtkWindowToImageFilter()
            writer = vtk.vtkPNGWriter()
            im.SetInput(render_window)
            im.Update()

            if if_save_img:
                writer.SetInputConnection(im.GetOutputPort())
                writer.SetFileName(save_path_without_suffix+'.png')
                writer.Write()

            if if_return_img:
                vtk_image = im.GetOutput()
                width, height, _ = vtk_image.GetDimensions()
                vtk_array = vtk_image.GetPointData().GetScalars()
                components = vtk_array.GetNumberOfComponents()

                arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
                arr = arr[::-1, :, :]
                return_dict['im'] = arr

            if if_save_obj:
                writer = vtk.vtkOBJExporter()
                writer.SetFilePrefix(save_path_without_suffix) # will be saved to .obj + .mtl to preserve color
                writer.SetInput(render_window)
                writer.Write()

        return return_dict
            

