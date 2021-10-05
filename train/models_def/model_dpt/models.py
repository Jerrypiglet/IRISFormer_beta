import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_misc import *

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
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


class DPT(BaseModel):
    def __init__(
        self,
        opt, 
        head,
        cfg_DPT, 
        features=256,
        backbone="vitb_rn50_384",
        if_imagenet_backbone=True, 
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        in_chans=3, 
        if_upscale_last_layer=True, 

    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last
        self.cfg_DPT = cfg_DPT

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
        self.pretrained, self.scratch = _make_encoder(
            cfg_DPT, 
            backbone,
            features,
            if_imagenet_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=self.output_hooks,
            use_vit_only=opt.cfg.MODEL_BRDF.DPT_baseline.use_vit_only, 
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
            in_chans=in_chans
        )

        self.scratch.refinenet1 = _make_fusion_block(opt, features, use_bn, if_upscale=if_upscale_last_layer)
        self.scratch.refinenet2 = _make_fusion_block(opt, features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(opt, features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(opt, features, use_bn)

        self.scratch.output_conv = head

        self.module_hooks_dict = {}

        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
            from models_def.model_dpt.utils_yogo import CrossAttention, LayerNormLastTwo

            module_dict = {}
            im_c = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.feat_proj_channels
            token_c = im_c
            if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer == 'instanceNorm':
                norm_layer_1d = nn.InstanceNorm1d
            elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer == 'layerNorm':
                norm_layer_1d = LayerNormLastTwo
                # assert False
            elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer == 'identity':
                norm_layer_1d = nn.Identity
            else:
                assert False, 'Invalid MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer'

            for layer_idx in range(len(self.pretrained.model.blocks)):
                module_dict['layer_%d_ca'%layer_idx] = CrossAttention(opt, token_c, im_c, token_c, norm_layer_1d=norm_layer_1d)

        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.keep_N_layers != -1:
            for layer_idx in range(len(self.pretrained.model.blocks)):
                if layer_idx >= self.num_layers:
                    self.pretrained.model.blocks[layer_idx] = nn.Identity()
                    if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
                        del module_dict['layer_%d_ca'%layer_idx]

        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
            self.ca_modules = nn.ModuleDict(module_dict)
            self.module_hooks_dict.update({'ca_modules': self.ca_modules, 'output_hooks': self.output_hooks})


    def forward(self, x, input_dict_extra={}):
        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)
        
        if self.cfg_DPT.if_share_pretrained:
            layer_1, layer_2, layer_3, layer_4 = input_dict_extra['shared_pretrained']
        else:
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.opt, self.cfg_DPT, self.pretrained, x, input_dict_extra={**input_dict_extra, **self.module_hooks_dict})

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        # print(self.scratch.layer1_rn, layer_1.shape, layer_1_rn.shape) # [hybrid] Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) torch.Size([2, 256, 64, 80]) torch.Size([2, 256, 64, 80])
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        # print(self.scratch.layer2_rn, layer_2.shape, layer_2_rn.shape) # [hybrid] Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) torch.Size([2, 512, 32, 40]) torch.Size([2, 256, 32, 40])
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        # print(self.scratch.layer3_rn, layer_3.shape, layer_3_rn.shape) # [hybrid] Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) torch.Size([2, 768, 16, 20]) torch.Size([2, 256, 16, 20])
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # print(self.scratch.layer4_rn, layer_4.shape, layer_4_rn.shape) # [hybrid] Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) torch.Size([2, 768, 8, 10]) torch.Size([2, 256, 8, 10])

        path_4 = self.scratch.refinenet4(layer_4_rn)
        # print(self.scratch.refinenet4, layer_4_rn.shape, path_4.shape)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        # print(self.scratch.refinenet3, layer_3_rn.shape, path_3.shape)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        # print(self.scratch.refinenet2, layer_2_rn.shape, path_2.shape)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # print(self.scratch.refinenet1, layer_1_rn.shape, path_1.shape)

        out = self.scratch.output_conv(path_1)

        return out

def get_BRDFNet_DPT(opt, model_path, backbone, modalities=[]):
    assert all([x in ['al', 'ro'] for x in modalities])

    module_dict = {}
    for modality in modalities:
        module_dict[modality] = DPTBRDFModel(
            opt=opt, 
            modality=modality, 
            path=model_path,
            backbone=backbone, 
            if_imagenet_backbone=opt.cfg.MODEL_BRDF.DPT_baseline.if_imagenet_backbone, 
            non_negative=True if opt.cfg.MODEL_BRDF.DPT_baseline.modality in ['de'] else False,
            enable_attention_hooks=opt.cfg.MODEL_BRDF.DPT_baseline.if_enable_attention_hooks,
            readout=opt.cfg.MODEL_BRDF.DPT_baseline.readout, 
            skip_keys=['scratch.output_conv'] if opt.cfg.MODEL_BRDF.DPT_baseline.if_skip_last_conv else [], 
            keep_keys=['pretrained.model.patch_embed.backbone'] if opt.cfg.MODEL_BRDF.DPT_baseline.if_only_restore_backbone else [], 
            cfg_DPT=opt.cfg.MODEL_BRDF.DPT_baseline
        )

    if opt.cfg.MODEL_BRDF.DPT_baseline.if_share_pretrained:
        module_dict['shared_pretrained'] = module_dict[modalities[0]].pretrained
        for modality in modalities:
            module_dict[modality].pretrained = nn.Identity()
    elif opt.cfg.MODEL_BRDF.DPT_baseline.if_share_patchembed:
        module_dict['shared_patch_embed_backbone'] = module_dict[modalities[0]].pretrained.model.patch_embed.backbone
        for modality in modalities:
            module_dict[modality].pretrained.model.patch_embed.backbone = nn.Identity()

    full_model = torch.nn.ModuleDict(module_dict)
    
    return full_model

class DPTBRDFModel(DPT):
    def __init__(
        self, opt, modality='al', path=None, non_negative=False, scale=1.0, shift=0.0, skip_keys=[], keep_keys=[], **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.modality = modality
        assert modality in ['al', 'de', 'ro'], 'Invalid modality: %s'%modality
        self.out_channels = {'al': 3, 'de': 1, 'ro': 1}[modality]

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
            print(magenta('===== [DPTBRDFModel] Loading %s'%path))
            self.load(path, skip_keys=skip_keys, keep_keys=keep_keys)
        # else:
        #     assert False, str(path)

    def forward(self, x, input_dict_extra={}):
        # print('[DPTBRDFModel - x]', x.shape, torch.max(x), torch.min(x), torch.median(x)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
        x_out = super().forward(x, input_dict_extra=input_dict_extra)
        # print('[DPTBRDFModel - x_out 1]', x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
        if self.modality == 'al':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
        elif self.modality == 'ro':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
            x_out = torch.mean(x_out, dim=1, keepdim=True)
        elif self.modality == 'de':
            '''
            where x_out is disparity (inversed * baseline)'''
            # x_out = torch.clamp(x_out, 1e-8, 100)
            # print('[DPTBRDFModel - x_out]', x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
            depth = self.scale * x_out + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            # x_out = torch.clip(x_out*5000., 1e-6, 2000000.)
            # print('[DPTBRDFModel - x_out 3]', x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out)) # torch.Size([1, 3, 288, 384]) tensor(1.3311, device='cuda:0', dtype=torch.float16) tensor(-1.0107, device='cuda:0', dtype=torch.float16) tensor(-0.4836, device='cuda:0', dtype=torch.float16)
            # pass

        return x_out, {}


def get_LightNet_DPT(opt, SGNum, model_path, backbone, modalities=[]):
    assert all([x in ['axis', 'lamb', 'weight'] for x in modalities])

    module_dict = {}
    for modality in modalities:
        module_dict[modality] = DPTLightModel(
            opt=opt, 
            SGNum=SGNum, 
            modality=modality, 
            path=model_path,
            backbone=backbone, 
            if_imagenet_backbone=opt.cfg.MODEL_LIGHT.DPT_baseline.if_imagenet_backbone, 
            enable_attention_hooks=opt.cfg.MODEL_BRDF.DPT_baseline.if_enable_attention_hooks,
            readout=opt.cfg.MODEL_LIGHT.DPT_baseline.readout, 
            skip_keys=['scratch.output_conv'] if opt.cfg.MODEL_BRDF.DPT_baseline.if_skip_last_conv else [], 
            keep_keys=['pretrained.model.patch_embed.backbone'] if opt.cfg.MODEL_BRDF.DPT_baseline.if_only_restore_backbone else [], 
            in_chans=opt.cfg.MODEL_LIGHT.DPT_baseline.in_channels, 
            if_upscale_last_layer=False, 
            cfg_DPT=opt.cfg.MODEL_LIGHT.DPT_baseline
        )

    if opt.cfg.MODEL_LIGHT.DPT_baseline.if_share_pretrained:
        module_dict['shared_pretrained'] = module_dict[modalities[0]].pretrained
        for modality in modalities:
            module_dict[modality].pretrained = nn.Identity()
    elif opt.cfg.MODEL_LIGHT.DPT_baseline.if_share_patchembed:
        module_dict['shared_patch_embed_backbone'] = module_dict[modalities[0]].pretrained.model.patch_embed.backbone
        for modality in modalities:
            module_dict[modality].pretrained.model.patch_embed.backbone = nn.Identity()

    full_model = torch.nn.ModuleDict(module_dict)
    
    return full_model

class DPTLightModel(DPT):
    def __init__(
        self, opt, SGNum,  modality, path=None, skip_keys=[], keep_keys=[], **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.SGNum = SGNum
        self.modality = modality
        assert modality in ['axis', 'lamb', 'weight'], 'Invalid modality: %s'%modality
        self.out_channels = {'axis': 3*SGNum, 'lamb': SGNum, 'weight': 3*SGNum}[modality]

        self.if_batch_norm = opt.cfg.MODEL_LIGHT.DPT_baseline.if_batch_norm

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
            # nn.ReLU(True) if non_negative else nn.Identity(),
        )
        # if self.if_batch_norm:
        #     head.insert(3, nn.GroupNorm(num_groups=4, num_channels=32))

        super().__init__(opt, head, **kwargs)

        if path is not None:
            print(magenta('===== [DPTLightModel] Loading %s'%path))
            self.load(path, skip_keys=skip_keys, keep_keys=keep_keys)
        # else:
        #     assert False, str(path)

    def forward(self, x, input_dict_extra={}):
        x_out = super().forward(x, input_dict_extra=input_dict_extra)
        # print(x_out.shape, torch.max(x_out), torch.min(x_out), torch.median(x_out))
        x_out = 1.01 * torch.tanh(x_out )

        if self.modality in ['lamb', 'weight']:
            x_out = 0.5 * (x_out + 1)
            # x_out = torch.clamp(x_out, 0., 1.)
        elif self.modality in ['axis']:
            bn, _, row, col = x_out.size()
            x_out = x_out.view(bn, self.SGNum, 3, row, col)
            x_out = x_out / torch.clamp(torch.sqrt(torch.sum(x_out * x_out,
                dim=2).unsqueeze(2) ), min = 1e-6).expand_as(x_out )
        else:
            assert False
        return x_out


# class DPTSegmentationModel(DPT):
#     def __init__(self, num_classes, path=None, **kwargs):

#         features = kwargs["features"] if "features" in kwargs else 256

#         kwargs["use_bn"] = True

#         head = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(True),
#             nn.Dropout(0.1, False),
#             nn.Conv2d(features, num_classes, kernel_size=1),
#             Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
#         )

#         super().__init__(head, **kwargs)

#         self.auxlayer = nn.Sequential(
#             nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(features),
#             nn.ReLU(True),
#             nn.Dropout(0.1, False),
#             nn.Conv2d(features, num_classes, kernel_size=1),
#         )

#         if path is not None:
#             self.load(path)
