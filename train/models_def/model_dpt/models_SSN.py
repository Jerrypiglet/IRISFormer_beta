import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_misc import *
import time

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    # _make_encoder,
    # forward_vit,
)

from .blocks_SSN import (
    # FeatureFusionBlock,
    # FeatureFusionBlock_custom,
    # Interpolate,
    _make_encoder_SSN,
    forward_vit_SSN,
    forward_vit_SSN_qkv_yogo
)

from models_def.model_dpt.utils_yogo import CrossAttention

class LayerNormLastTwo(nn.Module):
    def __init__(self, dim):
        super(LayerNormLastTwo, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, A):
        return torch.transpose(self.ln(torch.transpose(A, -1, -2)), -1, -2)


def _make_fusion_block(opt, features, use_bn, if_up_resize_override=None, if_assert_one_input=False):
    return FeatureFusionBlock_custom(
        opt, 
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        if_up_resize_override=if_up_resize_override, 
        if_assert_one_input=if_assert_one_input
    )


class DPT_SSN(BaseModel):
    def __init__(
        self,
        opt, 
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT_SSN, self).__init__()

        self.opt = opt

        # assert backbone in ['vitb_rn50_384', 'vitb_unet_384'], 'Backbone %s unsupported in DPT_SSN!'%backbone
        # assert backbone in ['vitb_unet_384', 'vitb16_384'], 'Backbone %s unsupported in DPT_SSN!'%backbone
        assert backbone in ['vitb_unet_384', 'vitb_unet_384_N_layer', 'vitl_unet_384'], 'Backbone %s unsupported in DPT_SSN!'%backbone

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitl_unet_384": [5, 11, 17, 23],
            "vitb_unet_384": [0, 1, 8, 11],
            "vitb_unet_384_N_layer": [opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.keep_N_layers-1]*4 if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.keep_N_layers!=-1 else [0, 0, 0, 0],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }
        self.hooks_backbone = hooks[backbone]

        vit_dims = {
            # "vitb_rn50_384": [0, 1, 8, 11],
            "vitl_unet_384": 1024,
            "vitb_unet_384": 768,
            "vitb_unet_384_N_layer": 768,
            # "vitb16_384": [2, 5, 8, 11],
            # "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder_SSN(
            opt, 
            backbone,
            features,
            use_pretrained=opt.cfg.MODEL_BRDF.DPT_baseline.if_imagenet_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_vit_only=opt.cfg.MODEL_BRDF.DPT_baseline.use_vit_only, 
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(opt, features, use_bn, if_up_resize_override=True)
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
            pass
        else:
            self.scratch.refinenet2 = _make_fusion_block(opt, features, use_bn)
            self.scratch.refinenet3 = _make_fusion_block(opt, features, use_bn)
            self.scratch.refinenet4 = _make_fusion_block(opt, features, use_bn, if_assert_one_input=True)
        
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
            # pass
            self.scratch.layer1_rn = nn.Identity()
            self.scratch.layer2_rn = nn.Identity()
            self.scratch.layer3_rn = nn.Identity()

        if backbone=='vitb_unet_384_N_layer':
            for layer_idx in range(len(self.pretrained.model.blocks)):
                if layer_idx >= opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.keep_N_layers:
                    self.pretrained.model.blocks[layer_idx] = nn.Identity()

        if not opt.cfg.MODEL_BRDF.DPT_baseline.if_pos_embed:
            self.pretrained.model.pos_embed = nn.Identity()

        self.scratch.output_conv = head

        self.recon_method = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_recon_method
        
        if self.recon_method == 'qkv':
            if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer == 'instanceNorm':
                norm_layer_1d = nn.InstanceNorm1d
            elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer == 'layerNorm':
                norm_layer_1d = LayerNormLastTwo
                # assert False
            elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer == 'identity':
                norm_layer_1d = nn.Identity
            else:
                assert False, 'Invalid MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer'
            module_dict = {}
            im_c = opt.cfg.MODEL_BRDF.DPT_baseline.feat_proj_channels
            token_c = vit_dims[backbone]
            assert im_c == token_c
            if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv:
                for layer_idx in range(12):
                    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_slim and layer_idx not in self.hooks_backbone:
                        continue

                    if backbone=='vitb_unet_384_N_layer' and layer_idx >= opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.keep_N_layers:
                        continue
                    # if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used and layer_idx!=self.hooks_backbone[-1]:
                    #     # print(module_dict['layer_%d_ca'%layer_idx].requires_grad, '-=--=-==')
                    #     with torch.no_grad():    
                    #         module_dict['layer_%d_ca'%layer_idx] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
                    #     continue
                        
                    module_dict['layer_%d_ca'%layer_idx] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)

            else:
                module_dict['layer_1_ca'] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
                module_dict['layer_2_ca'] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
                module_dict['layer_3_ca'] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
                module_dict['layer_4_ca'] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
            self.ca_modules = nn.ModuleDict(module_dict)

    def forward(self, x, input_dict_extra={}):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        if self.recon_method == 'qkv':
            input_dict_extra.update({'ca_modules': self.ca_modules})
        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv:
            layer_1, layer_2, layer_3, layer_4, ssn_return_dict = forward_vit_SSN_qkv_yogo(self.opt, self.pretrained, x, input_dict_extra=input_dict_extra, hooks=self.hooks_backbone)
        else:
            layer_1, layer_2, layer_3, layer_4, ssn_return_dict = forward_vit_SSN(self.opt, self.pretrained, x, input_dict_extra=input_dict_extra)

        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
            pass
        else:
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # print(layer_1_rn.shape, layer_2_rn.shape, layer_3_rn.shape, layer_4_rn.shape)
        # print(self.scratch.refinenet4)
        # print(self.scratch.refinenet3)
        # print(self.scratch.refinenet2)
        # print(self.scratch.refinenet1)

        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
            # path_3 = self.scratch.refinenet3(path_4)
            # path_2 = self.scratch.refinenet2(path_3)
            # path_1 = self.scratch.refinenet1(path_2)
            path_1 = self.scratch.refinenet1(layer_4_rn)
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_reduce_res:
        # print(path_4.shape, path_3.shape, path_2.shape, path_1.shape,)
        # else: (original)
        # torch.Size([2, 256, 16, 20]) torch.Size([2, 256, 32, 40]) torch.Size([2, 256, 64, 80]) torch.Size([2, 256, 128, 160])



        out = self.scratch.output_conv(path_1)

        return out, ssn_return_dict

class DPTAlbedoDepthModel_SSN(DPT_SSN):
    def __init__(
        self, opt, modality='al', path=None, non_negative=False, scale=1.0, shift=0.0, skip_keys=[], keep_keys=[], **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.modality = modality
        assert modality in ['al', 'de']
        self.out_channels = {'al': 3, 'de': 1}[modality]

        self.scale = scale
        self.shift = shift

        self.if_batch_norm = opt.cfg.MODEL_BRDF.DPT_baseline.if_batch_norm

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32) if self.if_batch_norm else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(32, self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(opt, head, **kwargs)

        if path is not None:
            print(magenta('===== [DPTAlbedoDepthModel_SSN] Loading %s'%path))
            self.load(path, skip_keys=skip_keys, keep_keys=keep_keys)
        # else:
        #     assert False, str(path)

    def forward(self, x, input_dict_extra={}):
        x_out, ssn_return_dict = super().forward(x, input_dict_extra=input_dict_extra)
        
        # if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone:
        #     print(ssn_return_dict['albedo_pred_unet'].shape)

        # print('[DPTAlbedoDepthModel_SSN - x_out 1]', x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
        if self.modality == 'al':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
            # print('[DPTAlbedoDepthModel_SSN - x_out 2]', self.if_batch_norm, x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
        # elif self.modality == 'de':
            # x_out = torch.clamp(x_out, 1e-8, 100)

        return x_out, ssn_return_dict