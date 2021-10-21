import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_misc import *

from .base_model import BaseModel
from .blocks_ViT import (
    _make_encoder_ViT,
    _make_decoder_ViT,
    forward_vit_ViT_encoder,
    forward_vit_ViT_decoder,
    MLP
)

class ViT(BaseModel):
    def __init__(
        self,
        opt, 
        cfg_ViT, 
        features=256,
        backbone="vitb_rn50_384_N_layers",
        if_imagenet_backbone=True, 
        channels_last=False,
        enable_attention_hooks=False,
        in_chans=3, 
        N_layers_encoder=6, 
        N_layers_decoder=6,
        head_names = [], 
        pool = 'cls' # pool strategy in the end: 'cls' or 'mean'
    ):

        super(ViT, self).__init__()

        self.channels_last = channels_last
        self.cfg_ViT = cfg_ViT

        self.opt = opt
        self.N_layers_encoder = N_layers_encoder
        self.N_layers_decoder = N_layers_decoder
        self.pool = pool
        self.head_names = head_names

        assert backbone == "vitb_rn50_384_N_layers"
        assert self.N_layers_encoder in [4, 6, 8]
        assert self.N_layers_decoder in [4, 6, 8]
        self.output_hooks_dict = {
            "4": [0, 1, 2, 3],
            "6": [0, 1, 3, 5],
            "8": [0, 1, 4, 7],
            # "12": [0, 1, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.encoder = _make_encoder_ViT(
            cfg_DPT = cfg_ViT, 
            backbone = backbone,
            use_pretrained = if_imagenet_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
            num_layers = int(self.N_layers_encoder), 
            hooks = self.output_hooks_dict[str(self.N_layers_encoder)],
            enable_attention_hooks = enable_attention_hooks,
            in_chans = in_chans
        )

        if cfg_ViT.if_share_decoder_over_heads:
            self.decoder = _make_decoder_ViT(
                cfg_DPT = cfg_ViT, 
                backbone = backbone,
                use_pretrained = if_imagenet_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
                num_layers = int(self.N_layers_decoder), 
                hooks = self.output_hooks_dict[str(self.N_layers_decoder)],
                enable_attention_hooks = enable_attention_hooks,
                in_chans = in_chans, 
            )
        else:
            assert self.head_names != []
            module_dict = {}
            for head_name in self.head_names:
                module_dict[head_name] = _make_decoder_ViT(
                    cfg_DPT = cfg_ViT, 
                    backbone = backbone,
                    use_pretrained = if_imagenet_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
                    num_layers = int(self.N_layers_decoder), 
                    hooks = self.output_hooks_dict[str(self.N_layers_decoder)],
                    enable_attention_hooks = enable_attention_hooks,
                    in_chans = in_chans, 
                )
            self.decoder = torch.nn.ModuleDict(module_dict)


        self.module_hooks_dict = {}

    def forward(self, x, input_dict_extra={}):
        # if self.channels_last:
        #     x.contiguous(memory_format=torch.channels_last)
        
        if self.cfg_ViT.if_share_encoder_over_modalities:
            _, _, _, layer_4_encoder = input_dict_extra['shared_encoder_outputs']
        else:
            _, _, _, layer_4_encoder = forward_vit_ViT_encoder(self.opt, self.cfg_ViT, self.encoder, x)
    
        # print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape) # torch.Size([16, 301, 768]) torch.Size([16, 301, 768]) torch.Size([16, 301, 768]) torch.Size([16, 301, 768])

        if self.cfg_ViT.if_share_decoder_over_heads:
            # x = self.decoder()
            _, _, _, layer_4_decoder = forward_vit_ViT_decoder(self.opt, self.cfg_ViT, self.decoder, layer_4_encoder)
            x = layer_4_decoder
            # print(x.shape)
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
            # print(x.shape)
            return x
        else:
            return_dict = {}
            for head_name in self.head_names:
                _, _, _, layer_4_decoder = forward_vit_ViT_decoder(self.opt, self.cfg_ViT, self.decoder[head_name], layer_4_encoder)
                x = layer_4_decoder
                # print(x.shape)
                x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
                # print(x.shape)
                return_dict[head_name] = x
            return return_dict


def get_LayoutNet_ViT(opt, backbone, N_layers_encoder, N_layers_decoder, modalities=[]):
    assert all([x in ['lo', 'ob'] for x in modalities])
    head_names_dict = {
        'lo': ['cam', 'lo'
            # 'pitch_reg', 
            # 'roll_reg', 
            # 'pitch_cls', 
            # 'roll_cls', 
            # 'lo_ori_reg', 
            # 'lo_ori_cls', 
            # 'lo_centroid', 
            # 'lo_coeffs', 
            ]
        }

    module_dict = {}
    for modality in modalities:
        module_dict[modality] = ViTLayoutModel(
            opt=opt, 
            modality=modality, 
            backbone=backbone, 
            if_imagenet_backbone=opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_imagenet_backbone, 
            # non_negative=True if opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.modality in ['de'] else False,
            # enable_attention_hooks=opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_enable_attention_hooks,
            # skip_keys=['scratch.output_conv'] if opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_skip_last_conv else [], 
            # keep_keys=['pretrained.model.patch_embed.backbone'] if opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_only_restore_backbone else [], 
            cfg_ViT=opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline, 
            N_layers_encoder=N_layers_encoder, 
            N_layers_decoder=N_layers_decoder, 
            head_names=head_names_dict[modality], 
            pool=opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.pool
        )

    if opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_share_encoder_over_modalities:
        module_dict['shared_encoder'] = module_dict[modalities[0]].encoder
        for modality in modalities:
            module_dict[modality].encoder = nn.Identity()
    # elif opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_share_patchembed:
    #     module_dict['shared_patch_embed_backbone'] = module_dict[modalities[0]].pretrained.model.patch_embed.backbone
    #     for modality in modalities:
    #         module_dict[modality].pretrained.model.patch_embed.backbone = nn.Identity()

    full_model = torch.nn.ModuleDict(module_dict)
    
    return full_model

class ViTLayoutModel(ViT):
    def __init__(
        self, 
        opt, 
        cfg_ViT, 
        modality='lo', 
        N_layers_encoder=6, 
        N_layers_decoder=6, 
        pool='cls', 
        head_names=[], 
        skip_keys=[], keep_keys=[], **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.modality = modality
        assert modality in ['lo', 'ob'], 'Invalid modality: %s'%modality
        assert pool in ['mean', 'cls']
        # self.out_channels = {'al': 3, 'de': 1, 'ro': 1}[modality]

        # head = nn.Sequential(
        #     nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32) if self.if_batch_norm else nn.Identity(),
        #     # nn.GroupNorm(num_groups=4, num_channels=32) if self.if_batch_norm else nn.Identity(),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, self.out_channels, kernel_size=1, stride=1, padding=0),
        #     # nn.BatchNorm2d(self.out_channels) if self.if_batch_norm else nn.Identity(),
        #     # nn.GroupNorm(self.out_channels) if self.if_batch_norm else nn.Identity(),
        #     nn.ReLU(True) if non_negative else nn.Identity(),
        #     nn.Identity(),
        # )
        # # if self.if_batch_norm:
        # #     head.insert(3, nn.GroupNorm(num_groups=4, num_channels=32))

        super().__init__(opt, cfg_ViT=cfg_ViT, head_names=head_names, pool=pool, N_layers_encoder=N_layers_encoder, N_layers_decoder=N_layers_decoder, **kwargs)

        if modality in ['lo']:
            if opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_indept_MLP_heads:
                self.heads = decoder_layout_emitter_heads_indeptMLP(opt, if_layout=True, 
                if_two_decoders=not opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_share_decoder_over_heads, 
                if_layer_norm=opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_indept_MLP_heads_if_layer_norm)
            else:
                self.heads = decoder_layout_emitter_heads(opt, if_layout=True, if_two_decoders=not opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_share_decoder_over_heads)
        else:
            assert False


    def forward(self, x, input_dict_extra={}):
        x_out = super().forward(x, input_dict_extra=input_dict_extra)
        if self.modality == 'lo':
            x_out = self.heads(x_out)
            
        return x_out

class decoder_layout_emitter_heads(nn.Module):
    def __init__(self, opt, if_layout, if_emitter_vanilla_fc=False, backbone_out_dim=768, if_two_decoders=False):
        super(decoder_layout_emitter_heads, self).__init__()
        self.opt = opt
        self.if_layout = if_layout
        self.if_emitter_vanilla_fc = if_emitter_vanilla_fc
        assert self.if_layout or self.if_emitter_vanilla_fc
        self.if_two_decoders = if_two_decoders

        self.embed_dim = backbone_out_dim
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ======== layout
        # '''Module parameters'''
        if self.if_layout:
            self.bin = opt.dataset_config.bins
            self.PITCH_BIN = len(self.bin['pitch_bin'])
            self.ROLL_BIN = len(self.bin['roll_bin'])
            self.LO_ORI_BIN = len(self.bin['layout_ori_bin'])

            # fc for camera
            self.fc_layout_1 = nn.Linear(self.embed_dim, self.embed_dim)
            self.fc_layout_2 = nn.Linear(self.embed_dim, (self.PITCH_BIN + self.ROLL_BIN) * 2)
            self.relu_layout_1 = nn.LeakyReLU(0.2, inplace=True)
            self.dropout_layout_1 = nn.Dropout(p=0.5)

            # fc for layout
            self.fc_layout_layout = nn.Linear(self.embed_dim, self.embed_dim)
            # for layout orientation
            self.fc_layout_3 = nn.Linear(self.embed_dim, self.embed_dim // 2)
            self.fc_layout_4 = nn.Linear(self.embed_dim // 2, self.LO_ORI_BIN * 2)
            # for layout centroid and coefficients
            self.fc_layout_5 = nn.Linear(self.embed_dim, self.embed_dim // 2)
            self.fc_layout_6 = nn.Linear(self.embed_dim // 2, 6)

        # ======== emitter
        if self.if_emitter_vanilla_fc:
            # fc for emitter ratio
            self.fc_emitter_1 = nn.Linear(self.embed_dim, self.embed_dim)
            self.relu_emitter_1 = nn.ReLU(inplace=True)
            self.fc_emitter_2 = nn.Linear(self.embed_dim, self.embed_dim // 2)
            self.relu_emitter_2 = nn.ReLU(inplace=True)

            # fc for emitter ratio: regress to area ratio
            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'wall_prob':
                self.cell_light_ratio = nn.Linear(self.embed_dim // 2, (opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size**2 + 1)*6)
            elif opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_prob':
                self.cell_light_ratio = nn.Linear(self.embed_dim // 2, (opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size**2)*6)
            elif opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
                self.cell_light_ratio = nn.Linear(self.embed_dim // 2, (opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size**2)*6)
            else:
                raise ValueError('Invalid: config.emitters.est_type')

            # fc for other emitter properties: cell_type, axis, intensity, lamb
            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type == 'cell_info':
                self.other_heads = torch.nn.ModuleDict({})
                for head_name, head_channels in [('cell_cls', 3), ('cell_axis', 3), ('cell_intensity', 3), ('cell_lamb', 1)]:
                    self.other_heads['fc_emitter_1_%s'%head_name] = nn.Linear(self.embed_dim, self.embed_dim)
                    self.other_heads['relu_emitter_1_%s'%head_name] = nn.ReLU(inplace=True)
                    self.other_heads['fc_emitter_2_%s'%head_name] = nn.Linear(self.embed_dim, self.embed_dim // 2)
                    self.other_heads['relu_emitter_2_%s'%head_name] = nn.ReLU(inplace=True)
                    self.other_heads['fc_emitter_3_%s'%head_name] = nn.Linear(self.embed_dim // 2, (opt.cfg.MODEL_LAYOUT_EMITTER.emitter.grid_size**2)*6 * head_channels)


    def forward(self, x):
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
        if self.if_layout:
            # branch for camera parameters
            cam = self.fc_layout_1(x['cam'] if self.if_two_decoders else x)
            cam = self.relu_layout_1(cam)
            cam = self.dropout_layout_1(cam)
            cam = self.fc_layout_2(cam)
            pitch_reg = cam[:, 0: self.PITCH_BIN]
            pitch_cls = cam[:, self.PITCH_BIN: self.PITCH_BIN * 2]
            roll_reg = cam[:, self.PITCH_BIN * 2: self.PITCH_BIN * 2 + self.ROLL_BIN]
            roll_cls = cam[:, self.PITCH_BIN * 2 + self.ROLL_BIN: self.PITCH_BIN * 2 + self.ROLL_BIN * 2]

            # branch for layout orientation, centroid and coefficients
            lo = self.fc_layout_layout(x['lo'] if self.if_two_decoders else x)
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

class decoder_layout_emitter_heads_indeptMLP(nn.Module):
    def __init__(self, opt, if_layout, if_emitter_vanilla_fc=False, backbone_out_dim=768, if_two_decoders=False, if_layer_norm=False):
        super(decoder_layout_emitter_heads_indeptMLP, self).__init__()
        self.opt = opt
        self.if_layout = if_layout
        self.if_emitter_vanilla_fc = if_emitter_vanilla_fc
        assert self.if_layout or self.if_emitter_vanilla_fc
        self.if_two_decoders = if_two_decoders

        self.embed_dim = backbone_out_dim
        
        # ======== layout
        # '''Module parameters'''
        if self.if_layout:
            self.bin = opt.dataset_config.bins
            self.PITCH_BIN = len(self.bin['pitch_bin'])
            self.ROLL_BIN = len(self.bin['roll_bin'])
            self.LO_ORI_BIN = len(self.bin['layout_ori_bin'])

            # fc for camera
            self.pitch_reg_MLP = MLP(self.embed_dim, self.embed_dim//2, self.PITCH_BIN, num_layers=3, if_layer_norm=if_layer_norm)
            self.pitch_cls_MLP = MLP(self.embed_dim, self.embed_dim//2, self.PITCH_BIN, num_layers=3, if_layer_norm=if_layer_norm)
            self.roll_reg_MLP = MLP(self.embed_dim, self.embed_dim//2, self.ROLL_BIN, num_layers=3, if_layer_norm=if_layer_norm)
            self.roll_cls_MLP = MLP(self.embed_dim, self.embed_dim//2, self.ROLL_BIN, num_layers=3, if_layer_norm=if_layer_norm)

            # fc for layout
            self.lo_ori_reg_MLP = MLP(self.embed_dim, self.embed_dim//2, self.LO_ORI_BIN, num_layers=3, if_layer_norm=if_layer_norm)
            self.lo_ori_cls_MLP = MLP(self.embed_dim, self.embed_dim//2, self.LO_ORI_BIN, num_layers=3, if_layer_norm=if_layer_norm)
            self.lo_centroid_MLP = MLP(self.embed_dim, self.embed_dim//2, 3, num_layers=3, if_layer_norm=if_layer_norm)
            self.lo_coeffs_MLP = MLP(self.embed_dim, self.embed_dim//2, 3, num_layers=3, if_layer_norm=if_layer_norm)

        # ======== emitter

    def forward(self, x):
        # ==== emitter & layout est
        return_dict_emitter  = {}

        # --- layout
        return_dict_layout = {}
        if self.if_layout:
            # branch for camera parameters
            x_cam = x['camera'] if self.if_two_decoders else x

            pitch_reg = self.pitch_reg_MLP(x_cam)
            pitch_cls = self.pitch_cls_MLP(x_cam)
            roll_reg = self.roll_reg_MLP(x_cam)
            roll_cls = self.roll_cls_MLP(x_cam)

            # branch for layout orientation, centroid and coefficients
            x_lo = x['layout'] if self.if_two_decoders else x

            lo_ori_reg = self.lo_ori_reg_MLP(x_lo)
            lo_ori_cls = self.lo_ori_cls_MLP(x_lo)
            lo_centroid = self.lo_centroid_MLP(x_lo)
            lo_coeffs = self.lo_coeffs_MLP(x_lo)

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
