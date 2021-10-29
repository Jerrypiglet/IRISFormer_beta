import struct
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_misc import *

from .base_model import BaseModel
# from .blocks import (
#     _make_encoder,
#     _make_scratch, 
#     forward_vit
# )
from .blocks_ViT import (
    _make_encoder_ViT,
    _make_decoder_ViT,
    forward_vit_ViT_encoder,
    forward_vit_ViT_decoder,
    forward_DPT_pretrained
    # _make_pretrained
)
# from .blocks import (
#     Interpolate,
# )
# from .vit_ViT import (
#     forward_flex_decoder
# )

# from .models import _make_fusion_block

# from .models_ViT import decoder_layout_emitter_heads_indeptMLP, decoder_layout_emitter_heads

from .models_DPT_BRDF import DPTBRDFModel
from .models_DPT_Light import DPTLightModel
from .models_ViT_layout import ViTLayoutObjModel

class ModelAll_ViT(torch.nn.Module):
    '''
    ViT/DPT for multiple modalities
    '''
    def __init__(
        self, 
        opt, 
        backbone, 
        N_layers_encoder_stage0, 
        N_layers_decoder_stage0, 
        N_layers_encoder_stage1, 
        N_layers_decoder_stage1, 
        modalities=[]):

        super(ModelAll_ViT, self).__init__()

        self.opt = opt

        head_names_dict = {
            # 'al': ['albedo'], 
            # 'no': ['normal'], 
            # 'de': ['depth'], 
            # 'ro': ['rough'], 
            'lo': ['camera', 'layout'], 
            # 'li': ['lighting']
            }
        # assert all([x in list(head_names_dict.keys()) for x in modalities])

        module_dict = {}
        for modality in modalities:
            if modality in ['lo']:
                module_dict[modality] = ViTLayoutObjModel(
                    opt=opt, 
                    cfg_ViT=opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline, 
                    modality=modality, 
                    backbone=backbone, 
                    if_imagenet_backbone=opt.cfg.MODEL_ALL.ViT_baseline.if_imagenet_backbone, 
                    if_share_encoder_over_modalities=opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage0, 
                    N_layers_encoder=N_layers_encoder_stage0, 
                    N_layers_decoder=N_layers_decoder_stage0, 
                    head_names=head_names_dict[modality], 
                    ViT_pool=opt.cfg.MODEL_ALL.ViT_baseline.ViT_pool
                )
            if modality in ['al', 'no', 'de', 'ro']:
                module_dict[modality] = DPTBRDFModel(
                    opt=opt, 
                    cfg_DPT=opt.cfg.MODEL_BRDF.DPT_baseline, 
                    modality=modality, 
                    backbone=backbone, 
                    features=256,
                    groups=1,
                    expand=False,
                    if_imagenet_backbone=opt.cfg.MODEL_BRDF.DPT_baseline.if_imagenet_backbone, 
                    if_share_encoder_over_modalities=opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage0, 
                    N_layers_encoder=N_layers_encoder_stage0, 
                    N_layers_decoder=N_layers_decoder_stage0, 
                    non_negative=True if modality in ['de'] else False,
                    # enable_attention_hooks=opt.cfg.MODEL_BRDF.DPT_baseline.if_enable_attention_hooks,
                    DPT_readout=opt.cfg.MODEL_BRDF.DPT_baseline.readout, 
                )
            if modality in ['axis', 'lamb', 'weight']:
                module_dict[modality] = DPTLightModel(
                    opt=opt, 
                    cfg_DPT=opt.cfg.MODEL_LIGHT.DPT_baseline, 
                    modality=modality, 
                    SGNum=opt.cfg.MODEL_LIGHT.SGNum, 
                    backbone=backbone, 
                    features=256,
                    groups=1,
                    expand=False,
                    if_imagenet_backbone=opt.cfg.MODEL_LIGHT.DPT_baseline.if_imagenet_backbone, 
                    if_share_encoder_over_modalities=opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage1, 
                    N_layers_encoder=N_layers_encoder_stage1, 
                    N_layers_decoder=N_layers_decoder_stage1, 
                    in_chans=opt.cfg.MODEL_LIGHT.DPT_baseline.in_channels, 
                    if_upscale_last_layer=False, 
                )
        
        self.modalities_stage0 = list(set(modalities) & set(['al', 'no', 'de', 'ro', 'lo']))
        if self.modalities_stage0 != []:
            if opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage0:
                module_dict['shared_encoder_stage0'] = module_dict[self.modalities_stage0[0]].encoder
                for modality in self.modalities_stage0:
                    module_dict[modality].encoder = nn.Identity()

        self.modalities_stage1 = list(set(modalities) & set(['axis', 'lamb', 'weight']))
        if self.modalities_stage1 != []:
            if opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage1:
                module_dict['shared_encoder_stage1'] = module_dict[self.modalities_stage1[0]].encoder
                for modality in self.modalities_stage1:
                    module_dict[modality].encoder = nn.Identity()

        if any([x in ['al', 'no', 'de', 'ro'] for x in modalities]):
            modalities_BRDF = list(set(['al', 'no', 'de', 'ro']) & set(modalities))
            if opt.cfg.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities:
                module_dict['shared_BRDF_decoder'] = module_dict[modalities_BRDF[0]].decoder
                for modality in modalities_BRDF:
                    module_dict[modality].decoder = nn.Identity()
            if opt.cfg.MODEL_ALL.ViT_baseline.if_share_pretrained_over_BRDF_modalities:
                assert opt.cfg.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities
                module_dict['shared_BRDF_pretrained'] = module_dict[modalities_BRDF[0]].pretrained
                self.unflatten = module_dict[modalities_BRDF[0]].unflatten
                for modality in modalities_BRDF:
                    module_dict[modality].pretrained = nn.Identity()

        self._ = torch.nn.ModuleDict(module_dict)
        
    def forward_brdf(self, x_stage0):
        input_dict_extra = {}

        if self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage0:
            input_dict_extra['shared_encoder_outputs'] = forward_vit_ViT_encoder(
                self.opt, self.opt.cfg.MODEL_BRDF.DPT_baseline, self._.shared_encoder_stage0, 
                x_stage0)

        if self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities:
            input_dict_extra['shared_BRDF_decoder_outputs'] = forward_vit_ViT_decoder(
                self.opt, self.opt.cfg.MODEL_BRDF.DPT_baseline, self._.shared_BRDF_decoder, 
                input_dict_extra['shared_encoder_outputs'][0])

        if self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_pretrained_over_BRDF_modalities:
            (_layer_1, _layer_2) = input_dict_extra['shared_encoder_outputs'][1]
            (_layer_3, _layer_4) = input_dict_extra['shared_BRDF_decoder_outputs'][1]
            input_dict_extra['shared_BRDF_pretrained_outputs'] = forward_DPT_pretrained(
                self.opt, self.opt.cfg.MODEL_BRDF.DPT_baseline, self._.shared_BRDF_pretrained, self.unflatten, 
                (_layer_1, _layer_2, _layer_3, _layer_4))

        output_dict = {}
        for modality in self.modalities_stage0:
            output_dict[modality] = self._[modality].forward(x_stage0, input_dict_extra=input_dict_extra)

        return output_dict

    def forward_light(self, x_stage1):
        input_dict_extra = {}

        assert self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage0
        input_dict_extra['shared_encoder_outputs'] = forward_vit_ViT_encoder(
            self.opt, self.opt.cfg.MODEL_LIGHT.DPT_baseline, self._.shared_encoder_stage1, 
            x_stage1)

        if self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities:
            input_dict_extra['shared_BRDF_decoder_outputs'] = forward_vit_ViT_decoder(
                self.opt, self.opt.cfg.MODEL_LIGHT.DPT_baseline, self._.shared_BRDF_decoder, 
                input_dict_extra['shared_encoder_outputs'][0])

        output_dict = {}
        for modality in self.modalities_stage1:
            output_dict[modality] = self._[modality].forward(None, input_dict_extra=input_dict_extra)

        return output_dict