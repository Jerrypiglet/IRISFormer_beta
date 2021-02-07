import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils.utils_total3D.utils_rui import Arrow3D, vis_axis, vis_cube_plt, vis_axis_xyz

import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from utils.utils_total3D.utils_OR_mesh import v_pairs_from_v3d_e
from utils.utils_total3D.utils_OR_cam import project_3d_line
from pathlib import Path
import matplotlib.patches as patches
from utils.utils_total3D.utils_OR_geo import bb_intersection_over_union
from utils.utils_total3D.utils_OR_geo import angle_between_2d

def RGB_to_01(RGB_tuple):
    return (float(RGB_tuple[0])/255., float(RGB_tuple[1])/255., float(RGB_tuple[2])/255.)

def shift_left(seq, n):
    return seq[n:]+seq[:n]

# https://stackoverflow.com/a/63625222
# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def read_vis_scene_3d(scene_dict, cam_dict=None, frame_dict=None, emitters_obj_list=None, \
    if_vis=False, if_vis_2d=True, if_vis_2dproj=True, if_vis_objs=True, if_vis_2dbbox=False, if_vis_emitters=True, if_indoor_objs_only=True, extra_transform_matrix=np.eye(3), cam_fromt_axis_id=2, zoom_2d=1., if_V2=True, \
    trans_3x3=None, if_reindex_layout=False):

    if trans_3x3 is None:
        trans_3x3 = np.eye(3, dtype=np.float32)
    
    # layout
    if 'v_skeleton' in scene_dict:
        v_skeleton , e_skeleton = scene_dict['v_skeleton'] @ trans_3x3.T, scene_dict['e_skeleton']
    layout_bbox_3d = scene_dict['layout_bbox_3d'] @ trans_3x3.T

    im_height, im_width = 480, 640
    ax_3d, ax_2d = None, None

    if if_vis:
        fig = plt.figure(figsize=(10, 10))
        ax_3d = fig.gca(projection='3d')
        ax_3d.set_proj_type('ortho')
        ax_3d.set_aspect("auto")
        if 'v_skeleton' in scene_dict:
            v_pairs = v_pairs_from_v3d_e(v_skeleton, e_skeleton)
            for v_pair in v_pairs:
                ax_3d.plot3D(v_pair[0], v_pair[1], v_pair[2])
        ax_3d.view_init(elev=-36, azim=89)
        vis_axis(ax_3d)
        vis_cube_plt(layout_bbox_3d, ax_3d, 'b', '--', if_face_idx_text=True, if_vertex_idx_text=True)

    if frame_dict is not None:
        if if_vis:
            if if_vis_2d:
                fig_2d = plt.figure(figsize=(15*zoom_2d, 8*zoom_2d))
                ax_2d = fig_2d.gca()
            im_uint8 = np.array(Image.open(frame_dict['frame_uint8_path']))
            im_height, im_width = im_uint8.shape[:2]

            if if_vis_2d:            
                ax_2d.imshow(im_uint8)

            if 'layout_wireframes_proj_list' in frame_dict:
                layout_wireframes_proj_list = frame_dict['layout_wireframes_proj_list']
            else:
                # project layout cuboid wireframes
                layout_wireframes_proj_list = []
                for idx, idx_list in enumerate([[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]):
                # for idx_list in [[3, 0]]:
                # for idx_list in [[5,6]]:
                    v3d_array = layout_bbox_3d
                    for i in range(len(idx_list)-1):
                        x1x2 = np.vstack((v3d_array[idx_list[i]], v3d_array[idx_list[i+1]]))
                        # print(cam_dict)
                        if 'cam_K' in cam_dict:
                            cam_K = cam_dict['cam_K']
                        else:
                            cam_K = np.array([[-cam_dict['f_px'], 0., im_width/2.], [0., -cam_dict['f_px'], im_height/2.], [0., 0., 1.]])
                        front_axis = (cam_dict['R_c'].T)[:, cam_fromt_axis_id:cam_fromt_axis_id+1]
                        x1x2_proj = project_3d_line(x1x2, cam_dict['R_c'], cam_dict['t_c'], cam_K, cam_dict['origin']+front_axis*0.01, front_axis, extra_transform_matrix=extra_transform_matrix)
                        layout_wireframes_proj_list.append(x1x2_proj)

            if if_vis_2d:
                for x1x2_proj in layout_wireframes_proj_list:
                    if x1x2_proj is not None:
                        ax_2d.plot([x1x2_proj[0][0], x1x2_proj[1][0]], [x1x2_proj[0][1], x1x2_proj[1][1]], color='r', linewidth=3)

                ax_2d.set_xlim([-im_width*0.5, im_width*1.5])
                ax_2d.set_ylim([im_height*1.5, -im_height*0.5])

        # plt.show()

    # camera
    if cam_dict is not None:
        if 'xaxis' not in cam_dict:
            R_c_inv = np.linalg.inv(cam_dict['R_c'])
            xaxis, yaxis, zaxis = np.split(R_c_inv, 3, axis=1)
            xaxis, yaxis, zaxis = trans_3x3 @ xaxis, trans_3x3 @ yaxis, trans_3x3 @ zaxis
        else:
            print('=====')
            xaxis, yaxis, zaxis = cam_dict['xaxis'], cam_dict['yaxis'], cam_dict['zaxis']
        # fov, f_px, im_width, im_height, R_c, t_c, origin, lookat, up = cam_dict['fov'], cam_dict['f_px'], cam_dict['width'], cam_dict['height'], cam_dict['R_c'], cam_dict['t_c'], cam_dict['origin'], cam_dict['lookat'], cam_dict['up']
        origin, lookat, up = trans_3x3 @ cam_dict['origin'], trans_3x3 @ cam_dict['lookat'], trans_3x3 @ cam_dict['up']
        # print(origin)
        # print(lookat)
        # print(up)
        # print(xaxis.shape)
        # if up is None:
        #     up = y_axis
        # # if lookat is None:
        # #     lookat = origin + x_axis
        # print(lookat, origin + zaxis)


        if if_vis:
            from utils.utils_rui import vis_axis_xyz
            vis_axis_xyz(ax_3d, xaxis.flatten(), yaxis.flatten(), zaxis.flatten(), origin.flatten(), suffix='_c')
            a = Arrow3D([origin[0][0], lookat[0][0]*2-origin[0][0]], [origin[1][0], lookat[1][0]*2-origin[1][0]], [origin[2][0], lookat[2][0]*2-origin[2][0]], mutation_scale=20,
                            lw=1, arrowstyle="->", color="b")
            ax_3d.add_artist(a)
            a_up = Arrow3D([origin[0][0], origin[0][0]+up[0][0]], [origin[1][0], origin[1][0]+up[1][0]], [origin[2][0], origin[2][0]+up[2][0]], mutation_scale=20,
                            lw=1, arrowstyle="->", color="r")
            ax_3d.add_artist(a_up)

    # reindex layout
    if if_reindex_layout:
        lo_center = np.mean(layout_bbox_3d[[0, 1, 2, 3], :], axis=0, keepdims=True)
        lo_center_vecs = layout_bbox_3d[[0, 1, 2, 3], :] - lo_center
        lo_center_vecs_2d = lo_center_vecs[:, [0, 2]]
        cam_dir_vec_2d = (lookat - origin)[[0, 2], :].reshape((2,))
        # angles = [angle_between((lookat - origin).reshape((3,)), x.reshape((3,))) / np.pi * 180. for x in lo_center_vecs] # angle from cam direction vector to the diagon (clockwise)
        angles = [angle_between_2d(x.reshape((2,)), cam_dir_vec_2d) for x in lo_center_vecs_2d] # angle from cam direction vector to the diagon (clockwise)
        angles_abs = [abs(x) for x in angles]
        smallest_idx = angles_abs.index(min(angles_abs)) 
        # new_idx_list1 = [abs(x-smallest_idx) % 4 for x in [0, 1, 2, 3]]
        new_idx_list1 = shift_left([0, 1, 2, 3], smallest_idx)
        new_idx_list2 = shift_left([4, 5, 6, 7], smallest_idx)
        # print(angles_abs, smallest_idx, new_idx_list1)
        layout_bbox_3d = np.vstack([layout_bbox_3d[new_idx_list1, :], layout_bbox_3d[new_idx_list2, :]])
        vis_cube_plt(layout_bbox_3d, ax_3d, 'None', '--', if_face_idx_text=True, if_vertex_idx_text=True, text_shift=[-0.5, 0., 0.], fontsize_scale=0.5)
        # pass
            
    # objects
    OR = 'OR45'
    if frame_dict is not None:
        vis_obj_random_ids = []
        # obj_bboxes_3d_list, obj_paths_list = scene_dict['obj_bboxes_3d_list'], scene_dict['obj_paths_list'], if_is_object, if_is_indoor_obj, obj_box_3d_proj, v_front_flags, v_inside_count, cat_id, cat_name

        for obj_idx, obj_dict in enumerate(frame_dict['obj_list']):
            if if_V2:
                cat_id, cat_name, cat_color = obj_dict['catInt_%s'%OR], obj_dict['catStr_%s'%OR], obj_dict['catColor_%s'%OR]
            else:
                cat_id, cat_name, cat_color = obj_dict['cat_id'], obj_dict['cat_name'], obj_dict['cat_color']

            # if_emitter = obj_dict['if_emitter'] and 'combined_filename' in obj_dict['emitter_prop']
            # if if_emitter:
            #     cat_name = cat_name + '**'

            # if obj_dict['cat_name'] == 'uv_mapped':
            if cat_id == 0:
                continue
            if if_indoor_objs_only and (not obj_dict['if_is_indoor_obj']):
                continue
            # cat_color = OR_mapping_id_to_color_dict[cat_id]
            # print(cat_name, cat_id, obj_path)
            # vis_cube_plt(obj_bbox_3d, ax, color=[float(x)/255. for x in cat_color], label=cat_name)
            if_visible = obj_dict['v_inside_count'] > 0
            if if_visible and 'random_id' in obj_dict:
                vis_obj_random_ids.append(obj_dict['random_id'])
            if if_vis and if_vis_objs:
                linestyle = '-' if if_visible else '--'
                if if_vis_2dproj and if_vis_2d:
                    v_2d_list, v_front_flags = obj_dict['obj_box_3d_proj'], obj_dict['v_front_flags']
                    for idx_list in [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
                        for i in range(len(idx_list)-1):
                            if v_front_flags[idx_list[i]] and v_front_flags[idx_list[i+1]]:
                                ax_2d.plot([v_2d_list[idx_list[i]][0], v_2d_list[idx_list[i+1]][0]], [v_2d_list[idx_list[i]][1], v_2d_list[idx_list[i+1]][1]], color=cat_color, linestyle=linestyle, linewidth=1)
                    if not if_vis_2dbbox:
                        ax_2d.text(v_2d_list[5][0], v_2d_list[5][1], cat_name, color=cat_color)

                vis_cube_plt(obj_dict['obj_box_3d'], ax_3d, cat_color, linestyle, cat_name)

                if if_vis_2dbbox and if_vis_2d:
                    bdb2d = {'x1': np.amin(obj_dict['obj_box_3d_proj'][:, 0]), 'x2': np.amax(obj_dict['obj_box_3d_proj'][:, 0]), 'y1': np.amin(obj_dict['obj_box_3d_proj'][:, 1]), 'y2': np.amax(obj_dict['obj_box_3d_proj'][:, 1])}
                    rect = patches.Rectangle((bdb2d['x1'], bdb2d['y1']), bdb2d['x2']-bdb2d['x1'], bdb2d['y2']-bdb2d['y1'], linewidth=1, edgecolor=cat_color, facecolor='none', linestyle='-.')
                    ax_2d.add_patch(rect)
                    image_bbox = [0., 0., im_width-1, im_height-1]
                    obj_bbox = [bdb2d['x1'], bdb2d['y1'], bdb2d['x2'], bdb2d['y2']]
                    iou, (interArea, boxAArea, boxBArea) = bb_intersection_over_union(image_bbox, obj_bbox, if_return_areas=True)
                    vis_ratio = interArea / (boxBArea+1e-5)
                    ax_2d.text(bdb2d['x1'], bdb2d['y1'], '%s-%.2f'%(cat_name, vis_ratio), color=cat_color)

        # ====== vis raw emitters_list in frame_dict (not transformed)
        if if_vis_emitters:
            if emitters_obj_list is not None:
                emitters_list_objs = emitters_obj_list
            else:
                emitters_list = frame_dict['emitters_list']
                # emitters_list = [x for x in emitters_list if 'id_random' in x and x['id_random'] in vis_obj_random_ids]
                for emitter_dict in emitters_list:
                    emitter_dict['if_vis'] = 'id_random' in emitter_dict and emitter_dict['id_random'] in vis_obj_random_ids
                assert emitters_list[0]['emitter_prop']['if_obj'] == False
                emitters_list_objs = emitters_list[1:]

            for obj_idx, obj_dict in enumerate(emitters_list_objs):
                if if_V2:
                    cat_id, cat_name, cat_color = obj_dict['catInt_%s'%OR], obj_dict['catStr_%s'%OR], obj_dict['catColor_%s'%OR]
                else:
                    cat_id, cat_name, cat_color = obj_dict['cat_id'], obj_dict['cat_name'], obj_dict['cat_color']

                # if_emitter = obj_dict['if_emitter'] and 'combined_filename' in obj_dict['emitter_prop']
                # if if_emitter:
                cat_name = cat_name + '**'

                # if obj_dict['cat_name'] == 'uv_mapped':
                if cat_id == 0:
                    continue
                if if_indoor_objs_only and (not obj_dict['if_is_indoor_obj']):
                    continue
                if if_vis:
                    # cat_color = OR_mapping_id_to_color_dict[cat_id]
                    # print(cat_name, cat_id, obj_path)
                    # vis_cube_plt(obj_bbox_3d, ax, color=[float(x)/255. for x in cat_color], label=cat_name)
                    # if_visible = obj_dict['v_inside_count'] > 0
                    if 'random_id' not in obj_dict:
                        if_visible = True
                    else:
                        if_visible = obj_dict['random_id'] in vis_obj_random_ids
                    # linestyle = '-' if if_visible else '--'
                    linestyle = '-.'
                    if if_vis_2dproj and if_vis_2d:
                        v_2d_list, v_front_flags = obj_dict['obj_box_3d_proj'], obj_dict['v_front_flags']
                        for idx_list in [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
                            for i in range(len(idx_list)-1):
                                if v_front_flags[idx_list[i]] and v_front_flags[idx_list[i+1]]:
                                    ax_2d.plot([v_2d_list[idx_list[i]][0], v_2d_list[idx_list[i+1]][0]], [v_2d_list[idx_list[i]][1], v_2d_list[idx_list[i+1]][1]], color=cat_color, linestyle=linestyle, linewidth=1)
                        if not if_vis_2dbbox:
                            ax_2d.text(v_2d_list[5][0], v_2d_list[5][1], cat_name, color=cat_color)

                    vis_cube_plt(obj_dict['obj_box_3d'] @ trans_3x3.T, ax_3d, cat_color, linestyle, cat_name)

                    if if_vis_2dbbox and if_vis_2d:
                        bdb2d = {'x1': np.amin(obj_dict['obj_box_3d_proj'][:, 0]), 'x2': np.amax(obj_dict['obj_box_3d_proj'][:, 0]), 'y1': np.amin(obj_dict['obj_box_3d_proj'][:, 1]), 'y2': np.amax(obj_dict['obj_box_3d_proj'][:, 1])}
                        rect = patches.Rectangle((bdb2d['x1'], bdb2d['y1']), bdb2d['x2']-bdb2d['x1'], bdb2d['y2']-bdb2d['y1'], linewidth=1, edgecolor=cat_color, facecolor='none', linestyle='-.')
                        ax_2d.add_patch(rect)
                        image_bbox = [0., 0., im_width-1, im_height-1]
                        obj_bbox = [bdb2d['x1'], bdb2d['y1'], bdb2d['x2'], bdb2d['y2']]
                        iou, (interArea, boxAArea, boxBArea) = bb_intersection_over_union(image_bbox, obj_bbox, if_return_areas=True)
                        vis_ratio = interArea / (boxBArea+1e-5)
                        ax_2d.text(bdb2d['x1'], bdb2d['y1'], '%s-%.2f'%(cat_name, vis_ratio), color=cat_color)
            # assert emitters_list[0]['emitter_prop']['if_obj'] == False

    if ax_3d is not None:
        ax_3d.set_box_aspect([1,1,1])
        set_axes_equal(ax_3d) # IMPORTANT - this is also required

    return ax_3d, ax_2d