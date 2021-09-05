import torch
import torch.nn as nn
import torch.nn.functional as F
from models_def.model_dpt.utils_yogo import CrossAttention_CAv2
from utils.utils_misc import *

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
)
from .blocks_CAv2 import (
    _make_encoder_CAv2,
    forward_vit_CAv2,
)


def _make_fusion_block(opt, features, use_bn, if_upscale=True):
    return FeatureFusionBlock_custom(
        opt, 
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        if_upscale=if_upscale
    )


class DPT_CAv2(BaseModel):
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

        super(DPT_CAv2, self).__init__()

        self.channels_last = channels_last

        self.opt = opt

        hooks_dict = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }
        num_layers_dict = {
            "vitb_rn50_384": 12,
            "vitb16_384": 12,
            "vitl16_384": 24,
        }

        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.keep_N_layers == -1:
            self.output_hooks = hooks_dict[backbone]
            self.num_layers = num_layers_dict[backbone]
        else:
            assert self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.keep_N_layers in [4, 8, 12]
            self.output_hooks = {
                "4": [0, 1, 2, 3],
                "8": [0, 1, 4, 7],
                "12": [0, 1, 8, 11],
            }[str(self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.keep_N_layers)]
            self.num_layers = self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.keep_N_layers

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder_CAv2(
            opt, 
            backbone,
            features,
            opt.cfg.MODEL_BRDF.DPT_baseline.if_imagenet_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=self.output_hooks,
            use_vit_only=opt.cfg.MODEL_BRDF.DPT_baseline.use_vit_only, 
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        if_upscale = not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_not_reduce_res
        self.scratch.refinenet1 = _make_fusion_block(opt, features, use_bn, if_upscale=True) # output layer
        self.scratch.refinenet2 = _make_fusion_block(opt, features, use_bn, if_upscale=if_upscale)
        self.scratch.refinenet3 = _make_fusion_block(opt, features, use_bn, if_upscale=if_upscale)
        self.scratch.refinenet4 = _make_fusion_block(opt, features, use_bn, if_upscale=if_upscale or opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_only_last_transformer_output_used)

        self.scratch.output_conv = head

        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_only_last_transformer_output_used:
            self.scratch.refinenet1 = nn.Identity()
            self.scratch.refinenet2 = nn.Identity()
            self.scratch.refinenet3 = nn.Identity()
        #     self.scratch.layer1_rn = nn.Identity()
        #     self.scratch.layer2_rn = nn.Identity()
        #     self.scratch.layer3_rn = nn.Identity()

        self.input_dict_extra = {}

        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer == 'instanceNorm':
            norm_layer_1d = nn.InstanceNorm1d
        elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer == 'layerNorm':
            norm_layer_1d = LayerNormLastTwo
            # assert False
        elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer == 'identity':
            norm_layer_1d = nn.Identity
        else:
            assert False, 'Invalid MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer'

        module_dict_ca = {}
        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA and not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA_if_grid_assembling:
            from models_def.model_dpt.utils_yogo import LayerNormLastTwo
            token_c = 768
            in_c = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.im_feat_init_c
            output_later_dims = [256, 512, 768, 768][::-1]
            out_c = in_c

            for layer_idx in range(len(self.pretrained.model.blocks)):
                if layer_idx in self.output_hooks:
                    out_c = output_later_dims.pop()
                module_dict_ca['layer_%d_ca'%layer_idx] = CrossAttention_CAv2(opt, token_c, input_dims=in_c, output_dims=out_c, norm_layer_1d=norm_layer_1d)
                # print(layer_idx, in_c, out_c)
                if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_init_img_feat:
                    pass
                else:
                    in_c = out_c

        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CAc:
            out_c = 768
            in_c = 768
            token_c = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.im_feat_init_c
            token_later_dims = [256, 512, 768, 768][::-1]

            for layer_idx in range(len(self.pretrained.model.blocks)-1):
                if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CAc_if_use_previous_feat:
                    # print(layer_idx, in_c, out_c, token_c, token_later_dims)
                    module_dict_ca['layer_%d_cac'%layer_idx] = CrossAttention_CAv2(opt, token_c, input_dims=in_c, output_dims=out_c, norm_layer_1d=norm_layer_1d)
                    if layer_idx in self.output_hooks:
                        token_c = token_later_dims.pop()
                elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CAc_if_use_init_feat:
                    module_dict_ca['layer_%d_cac'%layer_idx] = CrossAttention_CAv2(opt, token_c, input_dims=in_c, output_dims=out_c, norm_layer_1d=norm_layer_1d)
                else:
                    if layer_idx in self.output_hooks:
                        token_c = token_later_dims.pop()
                    # print(layer_idx, in_c, out_c, token_c, token_later_dims)
                    module_dict_ca['layer_%d_cac'%layer_idx] = CrossAttention_CAv2(opt, token_c, input_dims=in_c, output_dims=out_c, norm_layer_1d=norm_layer_1d)

        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.keep_N_layers != -1:
            for layer_idx in range(len(self.pretrained.model.blocks)):
                if layer_idx >= self.num_layers:
                    self.pretrained.model.blocks[layer_idx] = nn.Identity()
                    if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
                        if 'layer_%d_ca'%layer_idx in module_dict_ca:
                            del module_dict_ca['layer_%d_ca'%layer_idx]
                    if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CAc:
                        if 'layer_%d_cac'%layer_idx in module_dict_ca:
                            del module_dict_ca['layer_%d_cac'%layer_idx]

        for layer_idx in range(len(self.pretrained.model.blocks)):
            if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
                if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_init_img_feat:
                    if layer_idx not in self.output_hooks:
                        if 'layer_%d_ca'%layer_idx in module_dict_ca:
                            del module_dict_ca['layer_%d_ca'%layer_idx]
                        
        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
            self.ca_modules = nn.ModuleDict(module_dict_ca)
            self.input_dict_extra.update({'ca_modules': self.ca_modules, 'output_hooks': self.output_hooks})

    def forward(self, x, input_dict_extra={}):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4, output_dict_extra = forward_vit_CAv2(self.opt, self.pretrained, x, input_dict_extra={**self.input_dict_extra, **input_dict_extra})

        if_print = False

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        if if_print:
            print(layer_1.shape, layer_1_rn.shape) # [hybrid] Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) torch.Size([2, 256, 64, 80]) torch.Size([2, 256, 64, 80])
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        if if_print:
            print(layer_2.shape, layer_2_rn.shape) # [hybrid] Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) torch.Size([2, 512, 32, 40]) torch.Size([2, 256, 32, 40])
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if if_print:
            print(layer_3.shape, layer_3_rn.shape) # [hybrid] Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) torch.Size([2, 768, 16, 20]) torch.Size([2, 256, 16, 20])
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        if if_print:
            print(layer_4.shape, layer_4_rn.shape) # [hybrid] Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) torch.Size([2, 768, 8, 10]) torch.Size([2, 256, 8, 10])

        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_only_last_transformer_output_used:
            assert self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_not_reduce_res
            path_1 = self.scratch.refinenet4(layer_4_rn + layer_3_rn*0. + layer_2_rn*0. + layer_1_rn*0.) # [HACK] not using three other outputs while keeping their network layers
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn)
            if if_print:
                print(layer_4_rn.shape, path_4.shape)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            if if_print:
                print(layer_3_rn.shape, path_3.shape)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            if if_print:
                print(layer_2_rn.shape, path_2.shape)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
            if if_print:
                print(layer_1_rn.shape, path_1.shape)

        out = self.scratch.output_conv(path_1)

        return out, output_dict_extra


class DPTAlbedoDepthModel_CAv2(DPT_CAv2):
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
            # nn.GroupNorm(num_groups=4, num_channels=32) if self.if_batch_norm else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(32, self.out_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(self.out_channels) if self.if_batch_norm else nn.Identity(),
            # nn.GroupNorm(self.out_channels) if self.if_batch_norm else nn.Identity(),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        # if self.if_batch_norm:
        #     head.insert(3, nn.GroupNorm(num_groups=4, num_channels=32))

        super().__init__(opt, head, **kwargs)

        if path is not None:
            print(magenta('===== [DPTAlbedoDepthModel] Loading %s'%path))
            self.load(path, skip_keys=skip_keys, keep_keys=keep_keys)
        # else:
        #     assert False, str(path)

    def forward(self, x, input_dict_extra={}):
        # print('+++++++input_dict_extra', input_dict_extra.keys())
        # print('[DPTAlbedoDepthModel - x]', x.shape, torch.max(x), torch.min(x), torch.median(x)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
        x_out, output_dict_extra = super().forward(x, input_dict_extra=input_dict_extra)
        # print('[DPTAlbedoDepthModel - x_out 1]', x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
        if self.modality == 'al':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
            # print('[DPTAlbedoDepthModel - x_out 2]', self.if_batch_norm, x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
        elif self.modality == 'de':
            '''
            where x_out is disparity (inversed * baseline)'''
            # x_out = torch.clamp(x_out, 1e-8, 100)
            # print('[DPTAlbedoDepthModel - x_out]', x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
            depth = self.scale * x_out + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            # x_out = torch.clip(x_out*5000., 1e-6, 2000000.)
            # print('[DPTAlbedoDepthModel - x_out 3]', x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
            # pass

        return x_out, output_dict_extra