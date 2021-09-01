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
    _make_scratch_SSN_N_layers, 
    forward_vit_SSN,
    forward_vit_SSN_qkv_yogo, 
    forward_vit_SSN_qkv_yogo_N_layers
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


class DPT_SSN_yogoUnet_N_layers(BaseModel):
    def __init__(
        self,
        opt, 
        head,
        features=256,
        backbone="vitb_unet_384_N_layer",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT_SSN_yogoUnet_N_layers, self).__init__()

        self.opt = opt

        # assert backbone in ['vitb_rn50_384', 'vitb_unet_384'], 'Backbone %s unsupported in DPT_SSN!'%backbone
        # assert backbone in ['vitb_unet_384', 'vitb16_384'], 'Backbone %s unsupported in DPT_SSN!'%backbone
        assert backbone in ['vitb_unet_384_N_layer'], 'Backbone %s unsupported in DPT_SSN!'%backbone

        self.channels_last = channels_last
        N_layer_hooks_dict = {
            1: [0], 
            2: [0, 1], 
            4: [0, 1, 2, 3], 
            8: [0, 1, 4, 7], 
            12: [0, 1, 8, 11]
        }

        self.num_layers = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.keep_N_layers
        self.output_layers = N_layer_hooks_dict[self.num_layers] if not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used else [N_layer_hooks_dict[self.num_layers][-1]]
        self.hooks_backbone = N_layer_hooks_dict[self.num_layers]

        vit_dims = 768

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder_SSN(
            opt, 
            backbone,
            features,
            use_pretrained=opt.cfg.MODEL_BRDF.DPT_baseline.if_imagenet_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=self.output_layers,
            use_vit_only=opt.cfg.MODEL_BRDF.DPT_baseline.use_vit_only, 
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        refinenet_dict = {} # layer with smaller index is earlier
        for layer_idx in self.output_layers:
            # layer_idx = self.num_layers - 1 - i
            if_last_layer = layer_idx==self.output_layers[-1]
            if_first_layer = layer_idx==self.output_layers[0]
            refinenet_dict['%d'%layer_idx] = _make_fusion_block(opt, features, use_bn, if_up_resize_override=if_last_layer, if_assert_one_input=if_first_layer) # output layer: upsizex2 no matter if_not_reduce_res or not

        self.scratch.refinenet_dict = nn.ModuleDict(refinenet_dict)
            

        for layer_idx in range(len(self.pretrained.model.blocks)):
            if layer_idx >= self.num_layers:
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
            token_c = vit_dims
            assert im_c == token_c

            if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv:
                for layer_idx in range(self.num_layers):
                    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_slim and layer_idx not in self.hooks_backbone:
                        assert False, 'not support'
                        continue

                    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
                        if layer_idx >= self.num_layers:
                            continue
                    # else:
                    #     if layer_idx not in self.output_layers:
                    #         continue
                        
                    module_dict['layer_%d_ca'%layer_idx] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
                    # print('+++++++++', 'layer_%d_ca'%layer_idx)

            else:
                assert False, 'unsupported for now'
                # module_dict['layer_1_ca'] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
                # module_dict['layer_2_ca'] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
                # module_dict['layer_3_ca'] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
                # module_dict['layer_4_ca'] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)
            self.ca_modules = nn.ModuleDict(module_dict)

    def forward(self, x, input_dict_extra={}):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        if self.recon_method == 'qkv':
            input_dict_extra.update({'ca_modules': self.ca_modules})
        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv:
            layer_dict, ssn_return_dict = forward_vit_SSN_qkv_yogo_N_layers(self.opt, self.pretrained, x, input_dict_extra=input_dict_extra, hooks=self.output_layers)
        else:
            assert False
            # layer_1, layer_2, layer_3, layer_4, ssn_return_dict = forward_vit_SSN(self.opt, self.pretrained, x, input_dict_extra=input_dict_extra)

        # if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
        #     pass
        # else:
        #     if self.num_layers < 4:
        #         layer_2_rn = self.scratch.layer2_rn(layer_2)
        #         if self.num_layers < 3:
        #             layer_3_rn = self.scratch.layer3_rn(layer_3)
        #             if self.num_layers < 2:
        #                 layer_1_rn = self.scratch.layer1_rn(layer_1)

        # layer_4_rn = self.scratch.layer4_rn(layer_4)
        # # print(layer_1_rn.shape, layer_2_rn.shape, layer_3_rn.shape, layer_4_rn.shape)
        # # print(self.scratch.refinenet4)
        # # print(self.scratch.refinenet3)
        # # print(self.scratch.refinenet2)
        # # print(self.scratch.refinenet1)
        for layer_idx in self.output_layers:
            # print(layer_idx, self.scratch.layer_rn_dict['%d'%layer_idx])
            layer_dict['%d'%layer_idx] = self.scratch.layer_rn_dict['%d'%layer_idx](layer_dict['%d'%layer_idx])

        if len(self.output_layers)==1:
            layer_idx = self.output_layers[0]
            path_out = self.scratch.refinenet_dict['%d'%layer_idx](layer_dict['%d'%layer_idx])
        else:
            for layer_idx in self.output_layers:
                if layer_idx in [self.output_layers[0]]:
                    path_out = self.scratch.refinenet_dict['%d'%layer_idx](layer_dict['%d'%layer_idx])
                else:
                    path_out = self.scratch.refinenet_dict['%d'%layer_idx](path_out, layer_dict['%d'%layer_idx])

        out = self.scratch.output_conv(path_out)

        return out, ssn_return_dict

class DPTAlbedoDepthModel_SSN_yogoUnet_N_layers(DPT_SSN_yogoUnet_N_layers):
    '''
    N layers, 
    of which up to 4 layers are connected to outputs
    '''
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