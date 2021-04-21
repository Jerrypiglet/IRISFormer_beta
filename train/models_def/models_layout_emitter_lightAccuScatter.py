import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from models_def.models_light import renderingLayer
from icecream import ic
from utils.utils_misc import *

# V3 LightAccuNetScatter (not taking mean of accumulated light; scatter over a hemisphere centered at the emitter instead)
class decoder_layout_emitter_lightAccuScatter_UNet_V3(nn.Module):
    def __init__(self, opt=None, grid_size = 8, scatterHeight = 8, scatterWidth = 16, envmapFeatsChannels=128, other_flags=[]):
        super(decoder_layout_emitter_lightAccuScatter_UNet_V3, self).__init__()
        self.opt = opt
        self.grid_size = grid_size
        self.ngrids = 6 * self.grid_size * self.grid_size

        if self.opt is not None:
            assert self.opt.cfg.MODEL_LAYOUT_EMITTER.emitter.relative_dir == True

        self.use_weighted_axis = 'use_weighted_axis' in other_flags or self.opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_weighted_axis 

        self.scatterHeight = scatterHeight
        self.scatterWidth = scatterWidth

        self.envmapFeatsHeight = self.scatterHeight // 4
        self.envmapFeatsWidth = self.scatterWidth // 4
        self.envmapFeatsChannels = envmapFeatsChannels
        self.flattened_envmap_feats_channels = self.envmapFeatsChannels * self.envmapFeatsHeight * self.envmapFeatsWidth


        # envmap - encoders: envmap -> feats
        self.envmap_encoder_heads = torch.nn.ModuleDict({})
        for head_name, head_channels in [('cell_light_ratio', 1), ('cell_cls', 3), ('cell_axis', 3), ('cell_intensity', 3), ('cell_lamb', 1)]:
            self.envmap_encoder_heads[head_name] = self.get_envmap_encoder(head_name, out_channels=self.envmapFeatsChannels)

        # envmap - decoder (for cell_axis): feats -> weight
        if self.use_weighted_axis:
            self.envmap_decoder_heads = torch.nn.ModuleDict({})
            for head_name, head_channels in [('cell_axis', 3)]:
                self.envmap_decoder_heads[head_name] = self.get_envmap_decoder(head_name, in_channels=self.envmapFeatsChannels, out_channels=1)

        # UNet arch - encoders
        self.emitter_encoder_heads = torch.nn.ModuleDict({})
        for head_name, head_channels in [('cell_light_ratio', 1), ('cell_cls', 3), ('cell_axis', 3), ('cell_intensity', 3), ('cell_lamb', 1)]:
            self.get_emitter_encoder(head_name, in_channels=self.flattened_envmap_feats_channels)

        # UNet arch - decoders
        self.emitter_decoder_heads = torch.nn.ModuleDict({})

        self.UNet_decoder_heads_channels = [('cell_light_ratio', 1), ('cell_cls', 3), ('cell_intensity', 3), ('cell_lamb', 1)]
        if not self.use_weighted_axis:
            self.UNet_decoder_heads_channels.append(('cell_axis', 3))

        for head_name, head_channels in self.UNet_decoder_heads_channels:
            self.get_emitter_decoder(head_name, head_channels, in_channels=256)
        

    def get_envmap_encoder(self, head_name, out_channels):
        # same + downsize (4x8)
        envmap_encoder_heads = torch.nn.ModuleDict({})
        envmap_encoder_heads['conv1_%s_UNet'%head_name] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        envmap_encoder_heads['gn1_%s_UNet'%head_name] = nn.GroupNorm(num_groups=4, num_channels=64 )
        envmap_encoder_heads['conv2_%s_UNet'%head_name] = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding = 1, bias=True)
        envmap_encoder_heads['gn2_%s_UNet'%head_name] = nn.GroupNorm(num_groups=8, num_channels=128 )

        # same + downsize (2x4)
        envmap_encoder_heads['conv3_%s_UNet'%head_name] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True)
        envmap_encoder_heads['gn3_%s_UNet'%head_name] = nn.GroupNorm(num_groups=8, num_channels=128 )
        envmap_encoder_heads['conv4_%s_UNet'%head_name] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding = 1, bias=True)
        envmap_encoder_heads['gn4_%s_UNet'%head_name] = nn.GroupNorm(num_groups=8, num_channels=out_channels )

        return envmap_encoder_heads

    def get_envmap_decoder(self, head_name, in_channels, out_channels):
        # same + upsize (4x8)
        envmap_decoder_heads = torch.nn.ModuleDict({})
        envmap_decoder_heads['dconv1_%s_UNet'%head_name] = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True)
        envmap_decoder_heads['dgn1_%s_UNet'%head_name] = nn.GroupNorm(num_groups=8, num_channels=128 )
        envmap_decoder_heads['dconv2_%s_UNet'%head_name] = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        envmap_decoder_heads['dgn2_%s_UNet'%head_name] = nn.GroupNorm(num_groups=4, num_channels=64 )

        # same + upsize (8x16)
        envmap_decoder_heads['dconv3_%s_UNet'%head_name] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        envmap_decoder_heads['dgn3_%s_UNet'%head_name] = nn.GroupNorm(num_groups=8, num_channels=64 )
        envmap_decoder_heads['dconv4_%s_UNet'%head_name] = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        envmap_decoder_heads['dgn4_%s_UNet'%head_name] = nn.GroupNorm(num_groups=4, num_channels=64 )

        # same (8x16)
        envmap_decoder_heads['dconv5_%s_UNet'%head_name] = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding = 1, bias=True)


        return envmap_decoder_heads

    def get_emitter_encoder(self, head_name, in_channels=3):
        # same + downsize (4x4)
        self.emitter_encoder_heads['conv1_%s_UNet'%head_name] = nn.Conv2d(in_channels=in_channels*6, out_channels=64*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
        self.emitter_encoder_heads['gn1_%s_UNet'%head_name] = nn.GroupNorm(num_groups=4*6, num_channels=64*6 )
        self.emitter_encoder_heads['conv2_%s_UNet'%head_name] = nn.Conv2d(in_channels=64*6, out_channels=128*6, kernel_size=3, stride=2, padding = 1, bias=True, groups=6)
        self.emitter_encoder_heads['gn2_%s_UNet'%head_name] = nn.GroupNorm(num_groups=8*6, num_channels=128*6 )

        # same + downsize (2x2)
        self.emitter_encoder_heads['conv3_%s_UNet'%head_name] = nn.Conv2d(in_channels=128*6, out_channels=128*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
        self.emitter_encoder_heads['gn3_%s_UNet'%head_name] = nn.GroupNorm(num_groups=8*6, num_channels=128*6 )
        self.emitter_encoder_heads['conv4_%s_UNet'%head_name] = nn.Conv2d(in_channels=128*6, out_channels=256*6, kernel_size=3, stride=2, padding = 1, bias=True, groups=6)
        self.emitter_encoder_heads['gn4_%s_UNet'%head_name] = nn.GroupNorm(num_groups=16*6, num_channels=256*6 )


    def get_emitter_decoder(self, head_name, head_channels, in_channels=256):
        # same + upsize (4x4)
        self.emitter_decoder_heads['dconv1_%s_UNet'%head_name] = nn.Conv2d(in_channels=in_channels*6, out_channels=128*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
        self.emitter_decoder_heads['dgn1_%s_UNet'%head_name] = nn.GroupNorm(num_groups=8*6, num_channels=128*6 )
        self.emitter_decoder_heads['dconv2_%s_UNet'%head_name] = nn.Conv2d(in_channels=256*6, out_channels=64*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
        self.emitter_decoder_heads['dgn2_%s_UNet'%head_name] = nn.GroupNorm(num_groups=4*6, num_channels=64*6 )

        # same + upsize (8x8)
        self.emitter_decoder_heads['dconv3_%s_UNet'%head_name] = nn.Conv2d(in_channels=64*6, out_channels=64*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
        self.emitter_decoder_heads['dgn3_%s_UNet'%head_name] = nn.GroupNorm(num_groups=8*6, num_channels=64*6 )
        self.emitter_decoder_heads['dconv4_%s_UNet'%head_name] = nn.Conv2d(in_channels=128*6, out_channels=64*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)
        self.emitter_decoder_heads['dgn4_%s_UNet'%head_name] = nn.GroupNorm(num_groups=4*6, num_channels=64*6 )

        # same (8x8)
        self.emitter_decoder_heads['dconv5_%s_UNet'%head_name] = nn.Conv2d(in_channels=64*6, out_channels=head_channels*6, kernel_size=3, stride=1, padding = 1, bias=True, groups=6)

    def forward(self, scattered_light, emitter_outdirs_meshgrid_Total3D_outside):
        '''
        scattered_light: [B, ngrids, 8, 16, 3]
        '''
        assert len(scattered_light.shape)==5 and scattered_light.shape[1:]==(self.ngrids, self.scatterHeight, self.scatterWidth, 3)
        batch_size = scattered_light.shape[0]
        scattered_light_merged = scattered_light.reshape(-1, self.scatterHeight, self.scatterWidth, 3).permute(0, 3, 1, 2) # [B*ngrids, 3, 8, 16]

        # ======== get envmap features from scattered_light ========
        emitter_envmap_feats = {}
        for head_name, head_channels in [('cell_light_ratio', 1), ('cell_cls', 3), ('cell_axis', 3), ('cell_intensity', 3), ('cell_lamb', 1)]:
        # for head_name, head_channels in [('cell_light_ratio', 1)]:
            # xx = self.envmap_encoder_heads[head_name](scattered_light_merged)
            x = scattered_light_merged
            envmap_encoder_heads = self.envmap_encoder_heads[head_name]
            conv1, gn1 = envmap_encoder_heads['conv1_%s_UNet'%head_name], envmap_encoder_heads['gn1_%s_UNet'%head_name]
            conv2, gn2 = envmap_encoder_heads['conv2_%s_UNet'%head_name], envmap_encoder_heads['gn2_%s_UNet'%head_name]
            conv3, gn3 = envmap_encoder_heads['conv3_%s_UNet'%head_name], envmap_encoder_heads['gn3_%s_UNet'%head_name]
            conv4, gn4 = envmap_encoder_heads['conv4_%s_UNet'%head_name], envmap_encoder_heads['gn4_%s_UNet'%head_name]

            x1_8 = F.relu(gn1(conv1(x)), True) # [B*ngrids, 3, 8, 16]
            x2_4 = F.relu(gn2(conv2(x1_8)), True) # [B*ngrids, ?, 4, 8]
            x3_4 = F.relu(gn3(conv3(x2_4)), True) # [B*ngrids, ?, 4, 8]
            x4_2 = F.relu(gn4(conv4(x3_4)), True) # [B*ngrids, 128, 2, 4]

            x4_2_reshaped = x4_2.view(batch_size, self.ngrids, self.envmapFeatsChannels, self.envmapFeatsHeight, self.envmapFeatsWidth) # [B, ngrids, 128, 2, 4]

            emitter_envmap_feats[head_name] = x4_2_reshaped.view(batch_size, self.ngrids, self.flattened_envmap_feats_channels) # [B, ngrids, 128*2*4]

            if head_name == 'cell_axis':
                for layer_name, layer_feat in zip(['x1_8', 'x2_4', 'x3_4', 'x4_2'], [x1_8, x2_4, x3_4, x4_2]):
                    emitter_envmap_feats['%s-%s'%(head_name, layer_name)] = layer_feat


        return_dict_emitter = {'emitter_envmap_feats': emitter_envmap_feats, 'emitter_est_result':{}}

        # ======== get cell_axis results by weighted avg ========
        if self.use_weighted_axis:
            head_name = 'cell_axis'
            envmap_decoder_heads = self.envmap_decoder_heads[head_name]
            dconv1, dgn1 = envmap_decoder_heads['dconv1_%s_UNet'%head_name], envmap_decoder_heads['dgn1_%s_UNet'%head_name]
            dconv2, dgn2 = envmap_decoder_heads['dconv2_%s_UNet'%head_name], envmap_decoder_heads['dgn2_%s_UNet'%head_name]
            dconv3, dgn3 = envmap_decoder_heads['dconv3_%s_UNet'%head_name], envmap_decoder_heads['dgn3_%s_UNet'%head_name]
            dconv4, dgn4 = envmap_decoder_heads['dconv4_%s_UNet'%head_name], envmap_decoder_heads['dgn4_%s_UNet'%head_name]
            dconv5 = envmap_decoder_heads['dconv5_%s_UNet'%head_name]

            x1_8, x2_4, x3_4, x4_2 = [emitter_envmap_feats['%s-%s'%(head_name, layer_name)] for layer_name in ['x1_8', 'x2_4', 'x3_4', 'x4_2']]

            dx1_2 = F.relu(dgn1(dconv1(x4_2)), True)
            dx2_4_in = F.interpolate(dx1_2, scale_factor=2, mode='bilinear')
            dx2_4_in = torch.cat([dx2_4_in, x3_4], dim = 1)
            dx2_4 = F.relu(dgn2(dconv2(dx2_4_in)), True)

            dx3_4 = F.relu(dgn3(dconv3(dx2_4)), True)
            dx4_8_in = F.interpolate(dx3_4, scale_factor=2, mode='bilinear')
            dx4_8_in = torch.cat([dx4_8_in, x1_8], dim = 1)
            dx4_8 = F.relu(dgn4(dconv4(dx4_8_in)), True)

            dx5_8 = dconv5(dx4_8) # [768, 1, 8, 16]

            dx5_8_reshaped = dx5_8.view(batch_size, self.ngrids, 1, self.scatterHeight, self.scatterWidth) # [B, ngrids, 1, 8, 16]
            dx5_8_reshaped = dx5_8_reshaped.squeeze(2).view(batch_size, self.ngrids, -1) # [B, ngrids, 8*16]
            cell_axis_weights = torch.softmax(dx5_8_reshaped, -1).view(batch_size, self.ngrids, self.scatterHeight, self.scatterWidth, 1) # [B, ngrids, 8, 16, 1]
            assert len(emitter_outdirs_meshgrid_Total3D_outside.shape) == 5 and emitter_outdirs_meshgrid_Total3D_outside.shape == (batch_size, self.ngrids, self.scatterHeight, self.scatterWidth, 3)
            cell_axis_weighted = cell_axis_weights * emitter_outdirs_meshgrid_Total3D_outside # [B, ngrids, 8, 16, 3]
            cell_axis_weighted = torch.sum(torch.sum(cell_axis_weighted, 2), 2) # [B, ngrids, 3]
            cell_axis_out = cell_axis_weighted.view(batch_size, 6, self.grid_size, self.grid_size, 3) # in LightNet coords!!!!! Need to transform back to Total3D coords!!!

            # cell_axis_LightNet = cell_axis_out.unsqueeze(-2) @ transform_params_LightNet2Total3D['inv_post_transform_matrix_expand'] @ transform_params_LightNet2Total3D['inv_inv_cam_R_transform_matrix_pre_expand']
            # cell_axis_LightNet = cell_axis_LightNet.squeeze(-2)

            return_dict_emitter['emitter_est_result'].update({'cell_axis': cell_axis_out})

        # ======== get emitter est from U-Net for each estimated except `cell_axis` ========
        for head_name, head_channels in self.UNet_decoder_heads_channels:
            
            x = emitter_envmap_feats[head_name] # [B, ngrids, 128*2*4]
            x = x.permute(0, 2, 1) # [B, 128*2*4, ngrids]
            batch_size = x.shape[0]
            x = x.view(batch_size, self.flattened_envmap_feats_channels, 6, self.grid_size, self.grid_size)
            x = x.reshape(batch_size, self.flattened_envmap_feats_channels*6, self.grid_size, self.grid_size)

            # ---- emitter encoder
            conv1, gn1 = self.emitter_encoder_heads['conv1_%s_UNet'%head_name], self.emitter_encoder_heads['gn1_%s_UNet'%head_name]
            conv2, gn2 = self.emitter_encoder_heads['conv2_%s_UNet'%head_name], self.emitter_encoder_heads['gn2_%s_UNet'%head_name]
            conv3, gn3 = self.emitter_encoder_heads['conv3_%s_UNet'%head_name], self.emitter_encoder_heads['gn3_%s_UNet'%head_name]
            conv4, gn4 = self.emitter_encoder_heads['conv4_%s_UNet'%head_name], self.emitter_encoder_heads['gn4_%s_UNet'%head_name]

            x1_8 = F.relu(gn1(conv1(x)), True) # [2, 384, 8, 8]
            x2_4 = F.relu(gn2(conv2(x1_8)), True) # [2, 768, 4, 4]

            x3_4 = F.relu(gn3(conv3(x2_4)), True) # [2, 768, 4, 4]
            x4_2 = F.relu(gn4(conv4(x3_4)), True) # [2, 1536, 2, 2]

            # ---- emitter decoder
            dconv1, dgn1 = self.emitter_decoder_heads['dconv1_%s_UNet'%head_name], self.emitter_decoder_heads['dgn1_%s_UNet'%head_name]
            dconv2, dgn2 = self.emitter_decoder_heads['dconv2_%s_UNet'%head_name], self.emitter_decoder_heads['dgn2_%s_UNet'%head_name]
            dconv3, dgn3 = self.emitter_decoder_heads['dconv3_%s_UNet'%head_name], self.emitter_decoder_heads['dgn3_%s_UNet'%head_name]
            dconv4, dgn4 = self.emitter_decoder_heads['dconv4_%s_UNet'%head_name], self.emitter_decoder_heads['dgn4_%s_UNet'%head_name]
            dconv5 = self.emitter_decoder_heads['dconv5_%s_UNet'%head_name]

            dx1_2 = F.relu(dgn1(dconv1(x4_2)), True) # [2, 768, 2, 2]
            dx2_4_in = F.interpolate(dx1_2, scale_factor=2, mode='bilinear')
            dx2_4_in = torch.cat([dx2_4_in, x3_4], dim = 1) # [2, 1536, 4, 4]
            dx2_4 = F.relu(dgn2(dconv2(dx2_4_in)), True) # [2, 384, 4, 4]

            dx3_4 = F.relu(dgn3(dconv3(dx2_4)), True) # [2, 384, 4, 4]
            dx4_8_in = F.interpolate(dx3_4, scale_factor=2, mode='bilinear')
            dx4_8_in = torch.cat([dx4_8_in, x1_8], dim = 1) # [2, 768, 8, 8]
            dx4_8 = F.relu(dgn4(dconv4(dx4_8_in)), True)

            dx5_8 = dconv5(dx4_8)

            head_out = dx5_8.view(batch_size, 6, head_channels, self.grid_size, self.grid_size).permute(0, 1, 3, 4, 2) # [2, 6, head_channels, 8, 8] -> [2, 6, 8, 8, head_channels]
            return_dict_emitter['emitter_est_result'].update({head_name: head_out})

        return return_dict_emitter


    def print_net(self):
        count_grads = 0
        for name, param in self.named_parameters():
            if_trainable_str = white_blue('True') if param.requires_grad else green('False')
            print(name + str(param.shape) + if_trainable_str)
            if param.requires_grad:
                count_grads += 1
        print(magenta('---> ALL %d params; %d trainable'%(len(list(self.named_parameters())), count_grads)))
        return count_grads
