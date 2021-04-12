import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from models_def.models_light import renderingLayer

class decoder_layout_emitter_lightAccu(nn.Module):
    def __init__(self, opt=None, grid_size = 8, ):
        super(decoder_layout_emitter_lightAccu, self).__init__()
        self.opt = opt
        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(in_channels=3*6, out_channels=128*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
        self.gn1 = nn.GroupNorm(num_groups=8*6, num_channels=128*6 )

        self.conv2 = nn.Conv2d(in_channels=128*6, out_channels=256*6, kernel_size=5, stride=1, padding = 2, bias=True, groups=6)
        self.gn2 = nn.GroupNorm(num_groups=16*6, num_channels=256*6 )
        
        self.conv3 = nn.Conv2d(in_channels=256*6, out_channels=128*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
        self.gn3 = nn.GroupNorm(num_groups=8*6, num_channels=128*6 )


        self.decoder_heads = torch.nn.ModuleDict({})
        for head_name, head_channels in [('cell_light_ratio', 1), ('cell_cls', 3), ('cell_axis_global', 3), ('cell_intensity', 3), ('cell_lamb', 1)]:
            self.decoder_heads['decoder_conv2d_1_%s'%head_name] = nn.Conv2d(in_channels=128*6, out_channels=64*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
            self.decoder_heads['decoder_gn_1_%s'%head_name] = nn.GroupNorm(num_groups=4*6, num_channels=64*6 )
            self.decoder_heads['decoder_conv2d_2_%s'%head_name] = nn.Conv2d(in_channels=64*6, out_channels=head_channels*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
            # self.decoder_heads['decoder_gn_%s'%head_name] = nn.GroupNorm(num_groups=8*6, num_channels=head_channels*6 )
            # self.other_heads['fc_emitter_1_%s'%head_name] = nn.Linear(128, 1024)
            # self.other_heads['relu_emitter_1_%s'%head_name] = nn.ReLU(inplace=True)
            # self.other_heads['fc_emitter_2_%s'%head_name] = nn.Linear(1024, 512)
            # self.other_heads['relu_emitter_2_%s'%head_name] = nn.ReLU(inplace=True)
            # self.other_heads['fc_emitter_3_%s'%head_name] = nn.Linear(512, (opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size**2)*6 * head_channels)


    # def get_6_conv2d(self, layer_name='conv1', in_channels=3, out_channels=256, kernel_size=3, stride=1, padding = 1, bias=True, num_groups=16, num_channels=256 ):
    #     conv2d_dict = torch.nn.ModuleDict()
    #     for i in range(6):
    #         conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding = padding, bias=bias)
    #         gn = nn.GroupNorm(num_groups=16, num_channels=)
    #         conv2d_dict['%s_conv2d_%d'%(layer_name, i)] = conv
    #         conv2d_dict['%s_GH_%d'%(layer_name, i)] = gn



    def forward(self, envmap_lightAccu):
        assert len(envmap_lightAccu.shape)==5 and envmap_lightAccu.shape[1:]==(6, 3, self.grid_size, self.grid_size) # [B, 6, D(3), H(grid_size), W(grid_size)]
        batch_size = envmap_lightAccu.shape[0]
        envmap_lightAccu_merged = envmap_lightAccu.reshape(batch_size, -1, self.grid_size, self.grid_size) # [B, 6*D(3), H(grid_size), W(grid_size)]
        # print(envmap_lightAccu_merged.shape)

        x = envmap_lightAccu_merged
        # envmap_lightAccu_list = envmap_lightAccu.split([1, 1, 1, 1, 1, 1], dim=2)
        # print(len(envmap_lightAccu_list), envmap_lightAccu_list[0].shape)
        x = F.relu(self.gn1(self.conv1(x)), True) # torch.Size([2, 768, 8, 8])
        x = F.relu(self.gn2(self.conv2(x)), True) # torch.Size([2, 1536, 8, 8])
        x = F.relu(self.gn3(self.conv3(x)), True) # torch.Size([2, 768, 8, 8])

        return_dict_emitter = {}

        for head_name, head_channels in [('cell_light_ratio', 1), ('cell_cls', 3), ('cell_axis_global', 3), ('cell_intensity', 3), ('cell_lamb', 1)]:
            head_out = self.decoder_heads['decoder_gn_1_%s'%head_name](self.decoder_heads['decoder_conv2d_1_%s'%head_name](x))
            head_out = self.decoder_heads['decoder_conv2d_2_%s'%head_name](head_out)
            head_out = head_out.view(batch_size, 6, head_channels, self.grid_size, self.grid_size).permute(0, 1, 3, 4, 2) # [2, 6, head_channels, 8, 8] -> [2, 6, 8, 8, head_channels]
            # print(head_name, self.decoder_heads['decoder_conv2d_%s'%head_name](x).shape)
            return_dict_emitter.update({head_name: head_out})
            # print(head_name, head_out.shape)

        return {'emitter_est_result': return_dict_emitter}





class emitter_lightAccu(nn.Module):
    def __init__(self, opt=None, envHeight = 8, envWidth = 16, envRow = 120, envCol = 160, grid_size = 8, params=[]):
        super(emitter_lightAccu, self).__init__()
        self.opt = opt
        self.params = params

        self.envCol = envCol
        self.envRow = envRow
        self.envHeight = envHeight
        self.envWidth = envWidth

        self.grid_size = grid_size

        self.vv_envmap, self.uu_envmap = torch.meshgrid(torch.arange(self.envRow), torch.arange(self.envCol))
        self.vv_envmap, self.uu_envmap = self.vv_envmap.cuda(), self.uu_envmap.cuda()
        self.rL = renderingLayer(imWidth = envCol, imHeight = envRow)

        if self.opt is not None:
            self.im_width, self.im_height = self.opt.cfg.DATA.im_width, self.opt.cfg.DATA.im_height
        else:
            self.im_width, self.im_height = self.params['im_width'], self.params['im_height']
        self.u0_im = self.im_width / 2.
        self.v0_im = self.im_height / 2.

        self.vv_im, self.uu_im = torch.meshgrid(torch.arange(self.im_height), torch.arange(self.im_width))
        self.uu_im, self.vv_im = self.uu_im.cuda(), self.vv_im.cuda()

        basis_v_indexes = [(3, 2, 0), (7, 6, 4), (4, 5, 0), (6, 2, 5), (7, 6, 3), (7, 3, 4)]
        self.origin_v1_v2_list = [basis_v_indexes[wall_idx] for wall_idx in range(6)]
        ii, jj = torch.meshgrid(torch.arange(self.grid_size), torch.arange(self.grid_size))
        self.ii, self.jj = ii.cuda().unsqueeze(0).unsqueeze(0).unsqueeze(-1).float(), jj.cuda().unsqueeze(0).unsqueeze(0).unsqueeze(-1).float() # [B, 1, 8, 8, 1]
        self.extra_transform_matrix = torch.tensor([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]).unsqueeze(0).cuda().float() # [1, 3, 3]
        self.extra_transform_matrix_LightNet = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]).unsqueeze(0).cuda().float()


    def forward(self, input_dict):
        normalPred, depthPred, envmapsPredImage = input_dict['normalPred_lightAccu'], input_dict['depthPred_lightAccu'], input_dict['envmapsPredImage_lightAccu']
        cam_K, cam_R, layout = input_dict['cam_K'], input_dict['cam_R'], input_dict['layout'] # cam_R: cam axes, not transformation matrix!
        assert len(normalPred.shape) == 4 and normalPred.shape[1:] == (3, self.im_height, self.im_width)
        assert len(depthPred.shape) == 3 and depthPred.shape[1:] == (self.im_height, self.im_width)
        assert len(envmapsPredImage.shape) == 6 and envmapsPredImage.shape[1:] == (3, self.envRow, self.envCol, self.envHeight, self.envWidth)
        assert len(cam_K.shape) == 3 and cam_K.shape[1:] == (3, 3)
        assert len(layout.shape) == 3 and layout.shape[1:] == (8, 3)

        f_W_scale = self.im_width / 2. / cam_K[:, 0:1, 2:3]
        f_H_scale = self.im_height / 2. / cam_K[:, 1:2, 2:3]
        assert torch.max(torch.abs(f_W_scale - f_H_scale)) < 1e-4 # assuming f_x == f_y
        f_pix = cam_K[:, 0:1, 0:1] * f_W_scale # [B, 1, 1]
        z = - depthPred
        x = - (self.uu_im - self.u0_im) / f_pix * z
        y = (self.vv_im - self.v0_im) / f_pix * z
        points = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1) # (B, 3, 240, 320)

        ls_coords, camx, camy, normalPred = self.rL.forwardEnv(normalPred, envmapsPredImage, if_normal_only=True) # torch.Size([B, 128, 3, 120, 160]), [B, 3, 120, 160], [B, 3, 120, 160], [B, 3, 120, 160]


        verts_center_transformed_LightNet = self.get_grid_centers(layout, cam_R) # [B, 6, 8, 8, 3]

        envmap_lightAccu, points_sampled_mask_expanded = self.accu_light(points, verts_center_transformed_LightNet, camx, camy, normalPred, envmapsPredImage) # [B, 3, #grids, 120, 160]
        envmap_lightAccu_mean = (envmap_lightAccu.sum(-1).sum(-1) / (points_sampled_mask_expanded.sum(-1).sum(-1)+1e-6)).permute(0, 2, 1) # -> [1, 384, 3]

        
        return_dict = {'envmap_lightAccu': envmap_lightAccu, 'points_sampled_mask_expanded': points_sampled_mask_expanded, 'envmap_lightAccu_mean': envmap_lightAccu_mean}

        return return_dict

    def accu_light(self, points, verts_center_transformed_LightNet, camx, camy, normalPred, envmapsPredImage):
        batch_size = verts_center_transformed_LightNet.shape[0]
        p_t_all_grids = verts_center_transformed_LightNet.view(batch_size, -1, 3)
        ngrids = p_t_all_grids.shape[1]
        p_t_all_grids = p_t_all_grids.unsqueeze(2).unsqueeze(3) # torch.Size([B, #grids, 1, 1, 3])

        lightnet_downsample_ratio = self.im_height//self.envRow
        assert lightnet_downsample_ratio == self.im_width//self.envCol

        # points are in image shape (e.f. 240x320), while we need to sample per-lixel lighting points which at a lower res (e.g. 120x160).
        points_sampled = points[:, :, self.vv_envmap*lightnet_downsample_ratio, self.uu_envmap*lightnet_downsample_ratio].permute(0, 2, 3, 1) # [B, 120, 160, 3]
        points_sampled_mask = points_sampled[:, :, :, -1] < -0.1
        points_sampled = points_sampled.unsqueeze(1) # torch.Size([B, 1, 120, 160, 3])

        camx_, camy_, normalPred_ = camx[:, :, self.vv_envmap, self.uu_envmap], camy[:, :, self.vv_envmap, self.uu_envmap], normalPred[:, :, self.vv_envmap, self.uu_envmap]
        vec_to_t = p_t_all_grids - points_sampled # torch.Size([B, #grids, 120, 160, 3])

        A = torch.cat([camx_.permute(0, 2, 3, 1).unsqueeze(-1), camy_.permute(0, 2, 3, 1).unsqueeze(-1), normalPred_.permute(0, 2, 3, 1).unsqueeze(-1)], -1)
        eyes = torch.eye(3).float().cuda().reshape((1, 1, 1, 3, 3))
        A = A + 1e-6 * eyes
        AT = torch.inverse(A).unsqueeze(0) # -> [1, B, 120, 160, 3, 3]
        b = vec_to_t.unsqueeze(-1).transpose(0, 1) # [B, #grids, 120, 160, 3] -> [B, #grids, 120, 160, 3, 1]
        # print(AT.shape, b.shape)

        l_local = torch.matmul(AT, b) # ldirections (one point on the hemisphere; local) [#grids, B, 120, 160, 3, 1]
        l_local = l_local / torch.linalg.norm(l_local, dim=-2, keepdim=True) # [#grids, B, 120, 160, 3, 1]

        # l_local -> pixel coords in envmap
        cos_theta = l_local[:, :, :, :, 2, :]
        theta_SG = torch.arccos(cos_theta) # [0, pi] # el
        cos_phi = l_local[:, :, :, :, 0, :] / torch.sin(theta_SG)
        sin_phi = l_local[:, :, :, :, 1, :] / torch.sin(theta_SG)
        phi_SG = torch.atan2(sin_phi, cos_phi) # az
        # assert phi_SG >= -np.pi and phi_SG <= np.pi

        az_pix = (phi_SG / np.pi / 2. + 0.5) * self.envWidth - 0.5 # [#grids, B, 120, 160, 1]
        el_pix = theta_SG / np.pi * 2. * self.envHeight - 0.5 # [#grids, B, 120, 160, 1]
        ngrids = el_pix.shape[0]
        az_pix = az_pix.view(ngrids, -1, 1).transpose(0, 1) # [ngrids, B*120*160, 1] -> [B*120*160, ngrids, 1]
        el_pix = el_pix.view(ngrids, -1, 1).transpose(0, 1) # [ngrids, B*120*160, 1] -> [B*120*160, ngrids, 1]

        # # sample envmap
        # envmapsPredImage_ = envmapsPredImage[:, :, vv_envmap, uu_envmap, :, :] # [B, 3, 120, 160, 8, 16] -> [B, 3, 120, 160, 8, 16]
        envmapsPredImage_2 = envmapsPredImage.permute(0, 2, 3, 1, 4, 5).contiguous()
        envmapsPredImage_ = envmapsPredImage_2[:, self.vv_envmap, self.uu_envmap, :, :, :] # [B, 3, 120, 160, 8, 16] -> [B, 3, 120, 160, 8, 16] -> [B, 120, 160, 3, 8, 16]
        batch_size = envmapsPredImage_.shape[0]
        envmapsPredImage_ = envmapsPredImage_.view(-1, 3, 8, 16).unsqueeze(2)
        h, w = envmapsPredImage_.shape[-2:]
        uv_ = torch.cat([az_pix, el_pix], dim=-1) # [B*120*160, #grids, 2]
        uv_normalized = uv_ / (torch.tensor([w-1, h-1]).reshape(1, 1, 2).cuda().float()) * 2. - 1. # [B*120*160, #grids, 2]
        uv_normalized = uv_normalized.unsqueeze(2).unsqueeze(2)# -> [B*120*160, #grids, 1, 1, 2]
        uv_normalized = torch.cat([uv_normalized, torch.zeros_like(uv_normalized[:, :, :, :, 0:1])], dim=-1) # -> [B*120*160, #grids, 1, 1, 3]
        # essentially sample each 1x8x16 envmap at ngrid points, of B*120*160 envmaps
        # [B*120*160, 3, 1, 8, 16], [B*120*160, #grids, 1, 1, 3] -> [B*120*160, 3, #grids, 1, 1]
        envmap_lightAccu_ = torch.nn.functional.grid_sample(envmapsPredImage_, uv_normalized, padding_mode='border', align_corners = True) # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample

        # # print(envmapsPredImage.shape)
        # # print(envmapsPredImage.view(-1, 8, 16).shape)
        # # return vec_to_t, camx_, camy_, normalPred_, self.vv_envmap, self.uu_envmap, envmapsPredImage
        # # A = torch.cat([camx_.unsqueeze(2), camy_.unsqueeze(2), normalPred_.unsqueeze(2)], 2) # [B, 3, 3, 120, 160]
        # # batch_size = A.shape[0]
        # # AT = torch.inverse(A.permute(0, 3, 4, 1, 2).view(-1, 3, 3).unsqueeze(1).expand(-1, ngrids, -1, -1, -1, -1)) # [B, 3, 3, 120, 160] -> [B, 120, 160, 3, 3] -> [B, #grids, 120, 160, 3, 3]
        # # AT = AT.view(-1, 3, 3) # -> [?, 3, 3]
        # # b = vec_to_t.reshape(-1, 3).unsqueeze(-1) # [B, #grids, 120, 160, 3] -> [?, 3, 1]
        # A = torch.cat([camx_.permute(0, 2, 3, 1).unsqueeze(-1), camy_.permute(0, 2, 3, 1).unsqueeze(-1), normalPred_.permute(0, 2, 3, 1).unsqueeze(-1)], -1)
        # eyes = torch.eye(3).float().cuda().reshape((1, 1, 1, 3, 3))
        # A = A + 1e-6 * eyes
        # AT = torch.inverse(A).unsqueeze(1) # -> [B, 1, 120, 160, 3, 3]
        # b = vec_to_t.unsqueeze(-1) # [B, #grids, 120, 160, 3] -> [B, #grids, 120, 160, 3, 1]

        # l_local = torch.matmul(AT, b) # ldirections (one point on the hemisphere; local) [B, #grids, 120, 160, 3, 1]
        # l_local = l_local / torch.linalg.norm(l_local, dim=-2, keepdim=True) # [B, #grids, 120, 160, 3, 1]

        # # l_local -> pixel coords in envmap
        # cos_theta = l_local[:, :, :, :, 2, :]
        # theta_SG = torch.arccos(cos_theta) # [0, pi] # el
        # cos_phi = l_local[:, :, :, :, 0, :] / torch.sin(theta_SG)
        # sin_phi = l_local[:, :, :, :, 1, :] / torch.sin(theta_SG)
        # phi_SG = torch.atan2(sin_phi, cos_phi) # az
        # # assert phi_SG >= -np.pi and phi_SG <= np.pi

        # az_pix = (phi_SG / np.pi / 2. + 0.5) * self.rL.envWidth - 0.5 # [B, #grids, 120, 160, 1]
        # el_pix = theta_SG / np.pi * 2. * self.rL.envHeight - 0.5 # [B, #grids, 120, 160, 1]
        # az_pix = az_pix.permute(0, 2, 3, 1, 4).view(-1, ngrids, 1) # [B, #grids, 120, 160, 1] -> [B, 120, 160, ngrids, 1] -> [B*120*160, ngrids, 1]
        # el_pix = el_pix.permute(0, 2, 3, 1, 4).view(-1, ngrids, 1) # [B, #grids, 120, 160, 1] -> [B, 120, 160, ngrids, 1] -> [B*120*160, ngrids, 1]

        # # sample envmap
        # envmapsPredImage_ = envmapsPredImage[:, :, self.vv_envmap, self.uu_envmap, :, :].permute(0, 2, 3, 1, 4, 5) # [B, 3, 120, 160, 8, 16] -> [B, 3, 120, 160, 8, 16] -> [B, 120, 160, 3, 8, 16]
        # envmapsPredImage_ = envmapsPredImage_.view(-1, 3, 1, self.envHeight, self.envWidth)
        # h, w = envmapsPredImage_.shape[-2:]
        # uv_ = torch.cat([az_pix, el_pix], dim=-1) # [B*120*160, #grids, 2]
        # uv_normalized = uv_ / (torch.tensor([w-1, h-1]).reshape(1, 1, 2).cuda().float()) * 2. - 1. # [B*120*160, #grids, 2]
        # uv_normalized = uv_normalized.unsqueeze(2).unsqueeze(2)# -> [B*120*160, #grids, 1, 1, 2]
        # uv_normalized = torch.cat([uv_normalized, torch.zeros_like(uv_normalized[:, :, :, :, 0:1])], dim=-1) # -> [B*120*160, #grids, 1, 1, 3]
        # # print(envmapsPredImage_.shape, uv_normalized.shape)
        # # essentially sample each 1x8x16 envmap at ngrid points, of B*120*160 envmaps
        # # [B*120*160, 3, 1, 8, 16], [B*120*160, #grids, 1, 1, 3] -> [B*120*160, 3, #grids, 1, 1]
        # envmap_lightAccu_ = torch.nn.functional.grid_sample(envmapsPredImage_, uv_normalized, padding_mode='border', align_corners = True) # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample
        envmap_lightAccu_ = envmap_lightAccu_.squeeze(-1).squeeze(-1).view(batch_size, self.envRow, self.envCol, 3, ngrids) # -> [B, 120, 160, 3, #grids]
        # print(envmap_lightAccu_.shape)
        envmap_lightAccu_ = envmap_lightAccu_.permute(0, 3, 4, 1, 2) # -> [B, 3, #grids, 120, 160]
        points_sampled_mask_expanded = points_sampled_mask.float().unsqueeze(1).unsqueeze(1) # [B, 1, 1, 120, 160]
        envmap_lightAccu_ = envmap_lightAccu_ * points_sampled_mask_expanded
        envmap_lightAccu_ = envmap_lightAccu_ * 0.1
        # envmap_lightAccu_vis_ = torch.clip(envmap_lightAccu_**(1.0/2.2), 0., 1.)
        # envmap_lightAccu_vis_ = torch.clip(envmap_lightAccu_ * 0.5, 0., 1.) # [B, 3, #grids, 120, 160]

        # envmap_lightAccu_vis_mean_ = (envmap_lightAccu_vis_.sum(-1).sum(-1) / points_sampled_mask_expanded.sum(-1).sum(-1)).permute(0, 2, 1) # -> [1, 384, 3]
        # envmap_lightAccu_vis_max_ = envmap_lightAccu_vis_.amax(-1).amax(-1).permute(0, 2, 1) # -> [1, 384, 3]
        return envmap_lightAccu_, points_sampled_mask_expanded

    def get_grid_centers(self, layout, cam_R):
        basis_1_list = [(layout[:, origin_v1_v2[1], :] - layout[:, origin_v1_v2[0], :]) / self.grid_size for origin_v1_v2 in self.origin_v1_v2_list]
        basis_2_list = [(layout[:, origin_v1_v2[2], :] - layout[:, origin_v1_v2[0], :]) / self.grid_size for origin_v1_v2 in self.origin_v1_v2_list]
        origin_0_list = [layout[:, origin_v1_v2[0], :] for origin_v1_v2 in self.origin_v1_v2_list]

        basis_1_array = torch.stack(basis_1_list).permute(1, 0, 2).unsqueeze(2).unsqueeze(2).float() # [B, 6, 1, 1, 3]
        basis_2_array = torch.stack(basis_2_list).permute(1, 0, 2).unsqueeze(2).unsqueeze(2).float()
        origin_0_array = torch.stack(origin_0_list).permute(1, 0, 2).unsqueeze(2).unsqueeze(2).float()

        x_ij = basis_1_array * self.ii + basis_2_array * self.jj + origin_0_array # [B, 6, 8, 8, 3]
        x_i1j = basis_1_array * (self.ii+1.) + basis_2_array * self.jj + origin_0_array
        x_i1j1 = basis_1_array * (self.ii+1.) + basis_2_array * (self.jj+1.) + origin_0_array
        x_ij1 = basis_1_array * self.ii + basis_2_array * (self.jj+1.)+ origin_0_array
        verts_all = torch.stack([x_ij, x_i1j, x_i1j1, x_ij1], -1) # torch.Size([B, 6, 8, 8, 3, 4])
        # batch_size = verts_all.shape[0]
        # verts_flattened = verts_all.reshape(-1, 4, 3).transpose(-1, -2) # [B*6*8*8, 3, 4]

        cam_R_transform, cam_t_transform = cam_R.transpose(1, 2), torch.zeros((1, 1, 1, 1, 3, 1)).cuda().float() # R: cam axes -> transformation matrix
        # nverts = verts_flattened.shape[0]
        # cam_R_transform = cam_R_transform.unsqueeze(1).expand(-1, 6*self.grid_size*self.grid_size, -1, -1).view(-1, 3, 3) # [B, 3, 3] -> [B, 1, 3, 3] -> [B, 6*8*8, 3, 3] -> [B*6*8*8, 3, 3]
        cam_R_transform = cam_R_transform.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        x1x2_transformed = cam_R_transform @ verts_all + cam_t_transform
        x1x2_transformed = x1x2_transformed.transpose(-1, -2) @ (self.extra_transform_matrix.unsqueeze(1).unsqueeze(1).unsqueeze(1)) # camera projection coords: z forward, x right, y down
        verts_transformed_LightNet = x1x2_transformed @ (self.extra_transform_matrix_LightNet.unsqueeze(1).unsqueeze(1).unsqueeze(1)) # LightNet coords: z backward, x right, y up # [B, 6, 8, 8, 4, 3]
        # print(verts_transformed_LightNet.shape)

        verts_center_transformed_LightNet = torch.mean(verts_transformed_LightNet, 4) # [B, 6, 8, 8, 3]

        return verts_center_transformed_LightNet
