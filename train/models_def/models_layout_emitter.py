import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class decoder_layout_emitter(nn.Module):
    def __init__(self, opt):
        super(decoder_layout_emitter, self).__init__()
        self.opt = opt

        self.pad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn5 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad6 = nn.ZeroPad2d(1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn6 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad7 = nn.ZeroPad2d(1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, bias=True)
        self.gn7 = nn.GroupNorm(num_groups=64, num_channels=1024)

        backbone_out_dim = 1024
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ======== layout
        # '''Module parameters'''
        if 'lo' in self.opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
            bin = opt.dataset_config.bins
            self.PITCH_BIN = len(bin['pitch_bin'])
            self.ROLL_BIN = len(bin['roll_bin'])
            self.LO_ORI_BIN = len(bin['layout_ori_bin'])


            # fc for camera
            self.fc_layout_1 = nn.Linear(backbone_out_dim, 1024)
            self.fc_layout_2 = nn.Linear(1024, (self.PITCH_BIN + self.ROLL_BIN) * 2)
            self.relu_layout_1 = nn.LeakyReLU(0.2, inplace=True)
            self.dropout_layout_1 = nn.Dropout(p=0.5)

            # fc for layout
            self.fc_layout_layout = nn.Linear(backbone_out_dim, 1024)
            # for layout orientation
            self.fc_layout_3 = nn.Linear(1024, 512)
            self.fc_layout_4 = nn.Linear(512, self.LO_ORI_BIN * 2)
            # for layout centroid and coefficients
            self.fc_layout_5 = nn.Linear(1024, 512)
            self.fc_layout_6 = nn.Linear(512, 6)


        # ======== emitter
        self.if_emitter_vanilla_fc = self.opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.enable==False and 'em' in self.opt.cfg.MODEL_LAYOUT_EMITTER.enable_list
        if self.if_emitter_vanilla_fc:
            # fc for emitter ratio
            self.fc_emitter_1 = nn.Linear(backbone_out_dim, 1024)
            self.relu_emitter_1 = nn.ReLU(inplace=True)
            self.fc_emitter_2 = nn.Linear(1024, 512)
            self.relu_emitter_2 = nn.ReLU(inplace=True)

            # fc for emitter ratio: regress to area ratio
            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'wall_prob':
                self.cell_light_ratio = nn.Linear(512, (opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size**2 + 1)*6)
            elif opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_prob':
                self.cell_light_ratio = nn.Linear(512, (opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size**2)*6)
            elif opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
                self.cell_light_ratio = nn.Linear(512, (opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size**2)*6)
            else:
                raise ValueError('Invalid: config.emitters.est_type')

            # fc for other emitter properties: cell_type, axis, intensity, lamb
            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
                self.other_heads = torch.nn.ModuleDict({})
                for head_name, head_channels in [('cell_cls', 3), ('cell_axis', 3), ('cell_intensity', 3), ('cell_lamb', 1)]:
                    self.other_heads['fc_emitter_1_%s'%head_name] = nn.Linear(backbone_out_dim, 1024)
                    self.other_heads['relu_emitter_1_%s'%head_name] = nn.ReLU(inplace=True)
                    self.other_heads['fc_emitter_2_%s'%head_name] = nn.Linear(1024, 512)
                    self.other_heads['relu_emitter_2_%s'%head_name] = nn.ReLU(inplace=True)
                    self.other_heads['fc_emitter_3_%s'%head_name] = nn.Linear(512, (opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size**2)*6 * head_channels)


    def forward(self, input_feats_dict):

        # ==== backend & fc feats
        # for x in input_feats_dict:
        #     print(x, input_feats_dict[x].shape)
        # torch.Size([8, 3, 480, 640]) x5 torch.Size([8, 512, 15, 20])
        x = input_feats_dict['x4']
        # print(x.shape)
        x = F.relu(self.gn5(self.conv5(self.pad5(x))), True)
        # print(x.shape)
        x = F.relu(self.gn6(self.conv6(self.pad6(x))), True)
        # print(x.shape)
        x = F.relu(self.gn7(self.conv7(self.pad7(x))), True)
        # print(x.shape)
        x = self.avg_pool(x)
        # print(x.shape)
        x = x.reshape((x.shape[0], -1))

        # ==== emitter & layout est
        return_dict_emitter  = {}
        if self.if_emitter_vanilla_fc:
            # --- emitter
            cell_light_ratio = self.fc_emitter_1(x)
            cell_light_ratio = self.relu_emitter_1(cell_light_ratio)
            cell_light_ratio = self.fc_emitter_2(cell_light_ratio)
            cell_light_ratio = self.relu_emitter_2(cell_light_ratio)
            cell_light_ratio = self.cell_light_ratio(cell_light_ratio)
            return_dict_emitter = {'cell_light_ratio': cell_light_ratio}

            if self.opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
                for head_name, head_channels in [('cell_cls', 3), ('cell_axis', 3), ('cell_intensity', 3), ('cell_lamb', 1)]:
                    fc_out = self.other_heads['fc_emitter_1_%s'%head_name](x)
                    fc_out = self.other_heads['relu_emitter_1_%s'%head_name](fc_out)
                    fc_out = self.other_heads['fc_emitter_2_%s'%head_name](fc_out)
                    fc_out = self.other_heads['relu_emitter_2_%s'%head_name](fc_out)
                    fc_out = self.other_heads['fc_emitter_3_%s'%head_name](fc_out)
                    return_dict_emitter.update({head_name: fc_out})

        # --- layout
        return_dict_layout = {}
        if 'lo' in self.opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
            # branch for camera parameters
            cam = self.fc_layout_1(x)
            cam = self.relu_layout_1(cam)
            cam = self.dropout_layout_1(cam)
            cam = self.fc_layout_2(cam)
            pitch_reg = cam[:, 0: self.PITCH_BIN]
            pitch_cls = cam[:, self.PITCH_BIN: self.PITCH_BIN * 2]
            roll_reg = cam[:, self.PITCH_BIN * 2: self.PITCH_BIN * 2 + self.ROLL_BIN]
            roll_cls = cam[:, self.PITCH_BIN * 2 + self.ROLL_BIN: self.PITCH_BIN * 2 + self.ROLL_BIN * 2]

            # branch for layout orientation, centroid and coefficients
            lo = self.fc_layout_layout(x)
            lo = self.relu_layout_1(lo)
            lo = self.dropout_layout_1(lo)
            # branch for layout orientation
            lo_ori = self.fc_layout_3(lo)
            lo_ori = self.relu_layout_1(lo_ori)
            lo_ori = self.dropout_layout_1(lo_ori)
            lo_ori = self.fc_layout_4(lo_ori)
            lo_ori_reg = lo_ori[:, :self.LO_ORI_BIN]
            lo_ori_cls = lo_ori[:, self.LO_ORI_BIN:]

            # branch for layout centroid and coefficients
            lo_ct = self.fc_layout_5(lo)
            lo_ct = self.relu_layout_1(lo_ct)
            lo_ct = self.dropout_layout_1(lo_ct)
            lo_ct = self.fc_layout_6(lo_ct)
            lo_centroid = lo_ct[:, :3]
            lo_coeffs = lo_ct[:, 3:]

            # return pitch_reg, roll_reg, pitch_cls, roll_cls, lo_ori_reg, lo_ori_cls, lo_centroid, lo_coeffs
            return_dict_layout.update({
                'pitch_reg_result': pitch_reg, 
                'roll_reg_result': roll_reg, 
                'pitch_cls_result': pitch_cls, 
                'roll_cls_result': roll_cls, 
                'lo_ori_reg_result': lo_ori_reg, 
                'lo_ori_cls_result': lo_ori_cls, 
                'lo_centroid_result': lo_centroid, 
                'lo_coeffs_result': lo_coeffs
                })
            
        return_dict = {'layout_est_result': return_dict_layout, 'emitter_est_result': return_dict_emitter}

        return return_dict
