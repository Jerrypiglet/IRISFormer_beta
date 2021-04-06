import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils_SL_vis import vis_cube_plt, set_axes_equal, vis_axis, vis_axis_xyz, vis_index_map
from utils_SL_geo import get_front_3d_line, mask_for_polygons
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import torch
import time

class SimpleSceneTorch():

    def __init__(self, cam_dict, bbox):
        self.device = bbox.device        
        self.cam_params = self.form_camera(cam_dict)
        self.K = self.cam_params['K']
        self.T = torch.tensor([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]).to(self.device) # cam coords -> project cam coords; https://i.imgur.com/n8cpHe7.png
        self.H, self.W = cam_dict['height'], cam_dict['width']  
        self.bbox = bbox
        
        self.xyz_min = np.zeros(3,) + np.inf
        self.xyz_max = np.zeros(3,) - np.inf
        self.bbox_c = self.transform(self.bbox) # in camera coords
        self.bbox_c_T = (self.T @ (self.bbox_c.T)).T # in projection coords

        self.edge_list = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        self.face_list = [[1, 0, 2, 3], [4, 5, 7, 6], [0, 1, 4, 5], [1, 5, 2, 6], [3, 2, 7, 6], [4, 0, 7, 3]]
        self.edge_face_list = []
        for edge_vertices in self.edge_list:
            edge_tuple = (edge_vertices, [])
            for face_idx, face_vertices in enumerate(self.face_list):
                if edge_vertices[0] in face_vertices and edge_vertices[1] in face_vertices:
                    edge_tuple[1].append(face_idx)
            self.edge_face_list.append(edge_tuple)
    
    def form_camera(self, cam_dict):
        '''
        axes: np.array([[-1., 0., -1.], [0., 1., 0.], [1., 0., -1]])
        fov_x, fov_y: in degrees

        '''
        origin = cam_dict['origin'].reshape((3,1))
        if cam_dict['cam_axes'] is None:
            lookat_pnt = cam_dict['lookat'].reshape((3,1))
            toward = cam_dict['toward'].reshape((3,1))  # x-axis
            toward /= torch.linalg.norm(toward)
            up = cam_dict[6:9]  # y-axis
            up /= torch.linalg.norm(up)
            right = torch.cross(toward, up)  # z-axis
            right /= torch.linalg.norm(right)
        else:
            (toward, up, right) = torch.split(cam_dict['cam_axes'].T, 1, dim=1) # x_cam, y_cam, z_cam
            toward = toward / torch.linalg.norm(toward)
            up = up / torch.linalg.norm(up)
            right = right / torch.linalg.norm(right)
            assert abs(torch.dot(toward.flatten(), up.flatten())) < 1e-5
            assert abs(torch.dot(toward.flatten(), right.flatten())) < 1e-5
            assert abs(torch.dot(right.flatten(), up.flatten())) < 1e-5
            cam_axes = torch.hstack([toward, up, right]).T
            R = cam_axes.T  # columns respectively corresponds to toward, up, right vectors.
            t = origin

        width = cam_dict['width']
        height = cam_dict['height']
        if 'fov_x' in cam_dict and 'fov_y' in cam_dict:
            fov_x = cam_dict['fov_x'] / 180. * np.pi
            fov_y = cam_dict['fov_y'] / 180. * np.pi
            f_x = width / (2 * torch.tan(fov_x/2.))
            f_y = height / (2 * torch.tan(fov_y/2.))
        else:
            assert 'f_x' in cam_dict and 'f_y' in cam_dict
            f_x, f_y = cam_dict['f_x'], cam_dict['f_y']

        K = torch.tensor([[f_x, 0., (width-1)/2.], [0., f_y, (height-1)/2.], [0., 0., 1.]]).to(self.device)

        cam_params = {'K': K, 'R': R, 'origin': origin, 'cam_axes': cam_axes, 'toward': toward, 'up': up, 'right': right}
        cam_params.update({'f_x': f_x, 'f_y': f_y, 'u0': (width-1)/2., 'v0': (height-1)/2.})
        return cam_params

    def transform(self, x):
        assert len(x.shape)==2 and x.shape[1]==3
        x = x.reshape((-1, 3))
        return (self.cam_params['cam_axes'] @ (x.T - self.cam_params['origin'])).T

    def transform_and_proj(self, x):
        x_c = self.transform(x)
        x_c_T = (self.T @ (x_c.T)).T
        x_c_proj = (self.K @ x_c_T.T).T
        x_c_proj = x_c_proj[:, :2] / (x_c_proj[:, 2:3]+1e-6)
        front_flags = (x_c_T[:, 2]>0).tolist()
        return x_c_proj, front_flags

    def param_planes(self):
        plane_params = [[] for i in range(6)]
        vv, uu = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        uu, vv = uu.to(self.device), vv.to(self.device)
        invd_list = []

        for face_idx in range(6):
            # face_vertices = bbox[self.face_list[face_idx]]

            # https://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
            p1 = self.bbox_c_T[self.face_list[face_idx][0]]
            p2 = self.bbox_c_T[self.face_list[face_idx][1]]
            p3 = self.bbox_c_T[self.face_list[face_idx][2]]
            # These two vectors are in the plane
            v1 = p3 - p1
            v2 = p2 - p1
            # the cross product is a vector normal to the plane
            cp = torch.cross(v1, v2)
            a, b, c = cp
            # This evaluates a * x3 + b * y3 + c * z3 which equals d
            d = torch.dot(cp, p3)

            # print(face_idx, self.face_list[face_idx][:3],p1, p2, p3)
            # print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
            plane_params[face_idx] = [a, b, c, d]

            # Zhang et al. - 2020 - GeoLayout Geometry Driven Room Layout Estimation, Sec. 3.1
            p = -a / (self.cam_params['f_x'] * d)
            q = -b / (self.cam_params['f_y'] * d)
            r = 1/d * (a/self.cam_params['f_x']*self.cam_params['u0'] + b/self.cam_params['f_y']*self.cam_params['v0'] - c)
            invd = - (p * uu + q * vv + r)
                # print('>>>>>>>>', torch.sum(torch.isnan(invd)))
            invd_list.append(invd)
        return invd_list
        
    def vis_3d(self, bbox):
        fig = plt.figure(figsize=(5, 5))
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d = fig.gca(projection='3d')
        ax_3d.set_proj_type('ortho')
        ax_3d.set_aspect("auto")

        [cam_xaxis, cam_yaxis, cam_zaxis] = np.split(self.cam_params['cam_axes'].T, 3, 1)
        vis_cube_plt(ax_3d, bbox, linewidth=2, if_vertex_idx_text=True)
        vis_axis(ax_3d, make_bold=[1])
        vis_axis_xyz(ax_3d, cam_xaxis.flatten(), cam_yaxis.flatten(), cam_zaxis.flatten(), self.cam_params['origin'].flatten(), suffix='_c', make_bold=[0])

        self.xyz_min = np.minimum(self.xyz_min, np.amin(bbox, 0))
        self.xyz_max = np.maximum(self.xyz_max, np.amax(bbox, 0))
        self.xyz_min = np.minimum(self.xyz_min, self.cam_params['origin'].reshape((3,))-1.)
        self.xyz_max = np.maximum(self.xyz_max, self.cam_params['origin'].reshape((3,))+1.)
        ax_3d.view_init(elev=121, azim=-111)
        ax_3d.set_box_aspect([1,1,1])
        new_limits = np.hstack([self.xyz_min.reshape((3, 1)), self.xyz_max.reshape((3, 1))])
        set_axes_equal(ax_3d, limits=new_limits) # IMPORTANT - this is also required

        return ax_3d

    def vis_2d_bbox_proj(self, bbox, if_show=True, if_vertex_idx_text=True, edge_list=None):
        '''
        Projecting all vertices including those behind the camera; will cause artifacts: wrong locations of projected edges and edges behind the camera
        '''
        if edge_list is None:
            edge_idxes_list = self.edge_list
            verts = bbox
        else:
            verts = torch.vstack(edge_list)
            num_edges = len(edge_list)
            edge_idxes_list = [x.tolist() for x in np.split(np.arange(num_edges*2), num_edges)]
            if_vertex_idx_text = False

        verts_proj, front_flags = self.transform_and_proj(verts)
        # print(verts)
        # print(verts_proj)
        fig = plt.figure()
        for edge_idx, edge in enumerate(edge_idxes_list):
            x1 = verts_proj[edge[0]]
            x2 = verts_proj[edge[1]]
            # print(edge, x1, x2)
            plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color='k', linewidth=2, linestyle='--')
        if if_vertex_idx_text:
            for idx, x2d in enumerate(verts_proj):
                plt.text(x2d[0]+10, x2d[1]+10, str(idx))
        plt.axis('equal')
        plt.xlim([0., self.W-1])
        plt.ylim([self.H-1, 0])
        if if_show:
            plt.show()
        return plt.gca()

    def poly_to_masks(self, face_verts_list):
        mask_list = []
        mask_combined = torch.zeros(self.H, self.W, dtype=torch.long) + 6 # 6 for no faces, 0..5 for faces 0..5
        mask_conflict = np.zeros((self.H, self.W), np.bool)

        for face_idx, face_verts in enumerate(face_verts_list):
            if len(face_verts)==0:
                continue
            face_verts_proj = self.transform_and_proj(face_verts)
            face_verts_proj_reindex = ConvexHull(face_verts_proj[0].cpu().numpy()).vertices
            face_verts_proj_convex = face_verts_proj[0][face_verts_proj_reindex]

            # reduce poly to screen space to speed up rasterization
            p1 = Polygon([x.tolist() for x in face_verts_proj_convex])
            p2 = Polygon([(0, 0), (self.W-1, 0.), (self.W-1,self.H-1), (0, self.H-1)])
            face_poly = p1.intersection(p2)
            
            if face_poly.is_empty:
                continue
            mask = mask_for_polygons([face_poly], (self.H, self.W))
            mask = mask == 1
            mask_conflict = np.logical_or(mask_conflict, np.logical_and((mask_combined!=6).cpu().numpy(), mask))
            mask = np.logical_and(mask, np.logical_not(mask_conflict))
            mask_combined[mask] = face_idx
            mask_list.append((face_idx, mask))

        return mask_combined, [[x[0], np.logical_and(np.logical_not(mask_conflict), x[1])] for x in mask_list], mask_conflict
    
    def vis_mask_combined(self, mask_combined, ax_2d=None):
        # print(ax_2d)
        assert ax_2d is not None
        index_map_vis = vis_index_map(mask_combined)
        ax_2d.imshow(index_map_vis)
        return ax_2d


    def get_edges_front(self, ax_3d=None, if_vis=False):
        edges_front_list = []
        face_edges_list = [[] for i in range(6)]
        face_verts_list = [[] for i in range(6)]
        for edge_idx, (edge, edge_face) in enumerate(zip(self.edge_list, self.edge_face_list)):
            x1x2 = self.bbox[edge]
            _, front_flags = self.transform_and_proj(x1x2)
            x1x2_front, if_new_tuple = get_front_3d_line(x1x2, front_flags, self.cam_params['origin'], self.cam_params['toward']*0.01, if_torch=True)
            if x1x2_front is not None:
                # print('----', edge, x1x2, x1x2_front)
                edges_front_list.append((x1x2_front, if_new_tuple))
                edge_face_face_idxes = edge_face[1]
                for face_idx in edge_face_face_idxes:
                    face_edges_list[face_idx].append((x1x2_front, if_new_tuple))
        edges_front = torch.stack([x[0] for x in edges_front_list])

        if if_vis:
            for edge_front in edges_front:
                ax_3d.plot3D(edge_front[:, 0], edge_front[:, 1], edge_front[:, 2], color='r', linestyle='-', linewidth=3)

        new_edge_list = []
        for face_idx, face_edges in enumerate(face_edges_list):
            if len(face_edges)==0:
                continue
            all_verts = torch.vstack([x[0] for x in face_edges])
            face_verts_list[face_idx] = torch.unique(all_verts.detach(), dim=0)

            if if_vis:
                all_verts_if_new = torch.stack([x[1] for x in face_edges]).flatten()
                new_edge = all_verts[all_verts_if_new]
                if new_edge.shape[0]!=0:
                    new_edge = torch.unique(new_edge.detach(), dim=0)
                    assert new_edge.shape==(2, 3)
                    new_edge_list.append(new_edge)
                    ax_3d.plot3D(new_edge[:, 0], new_edge[:, 1], new_edge[:, 2], color='m', linestyle='--', linewidth=3)

        return edges_front_list, face_edges_list, face_verts_list



