import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_misc import *

from .base_model import BaseModel
from .blocks import (
    _make_encoder,
    _make_scratch, 
    forward_vit
)
from .blocks_ViT import (
    _make_encoder_ViT,
    _make_decoder_ViT,
    forward_vit_ViT,
)
from .blocks import (
    Interpolate,
)

from .models_ViT import decoder_layout_emitter_heads_indeptMLP, decoder_layout_emitter_heads

class Transformer_Hybrid_Encoder_Decoder(BaseModel):
    def __init__(
        self,
        opt, 
        cfg_ViT, 
        head_names = [], 
        features=256,
        backbone="vitb_rn50_384_N_layers",
        if_imagenet_backbone=True,
        if_share_encoder_over_modalities=True, 
        channels_last=False,
        enable_attention_hooks=False,
        in_chans=3, 
        N_layers_encoder=6, 
        N_layers_decoder=6,
        ViT_pool = 'mean',  # ViT_pool strategy in the end: 'cls' or 'mean'
        DPT_if_upscale_last_layer=True, 
    ):

        super(Transformer_Hybrid_Encoder_Decoder, self).__init__()

        self.channels_last = channels_last
        self.cfg_ViT = cfg_ViT

        self.opt = opt
        self.N_layers_encoder = N_layers_encoder
        self.N_layers_decoder = N_layers_decoder
        self.ViT_pool = ViT_pool
        self.head_names = head_names
        self.if_share_encoder_over_modalities = if_share_encoder_over_modalities
        
        assert backbone == "vitb_rn50_384_N_layers"
        assert self.N_layers_encoder in [4, 6, 8]
        assert self.N_layers_decoder in [4, 6, 8]
        # self.output_hooks_dict = {
        #     "4": [0, 1, 2, 3],
        #     "6": [0, 1, 3, 5],
        #     "8": [0, 1, 4, 7],
        # }
        self.output_hooks_dict = {
            "6": [[0, 1], [3, 5]],
        }

        # Instantiate backbone and reassemble blocks
        self.encoder = _make_encoder_ViT(
            cfg_DPT = cfg_ViT, 
            backbone = backbone,
            use_pretrained = if_imagenet_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
            num_layers = int(self.N_layers_encoder), 
            hooks = self.output_hooks_dict[str(self.N_layers_encoder)][0],
            enable_attention_hooks = enable_attention_hooks,
            in_chans = in_chans
        )

        if cfg_ViT.if_share_decoder_over_heads:
            self.decoder = _make_decoder_ViT(
                cfg_DPT = cfg_ViT, 
                backbone = backbone,
                use_pretrained = if_imagenet_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
                num_layers = int(self.N_layers_decoder), 
                hooks = self.output_hooks_dict[str(self.N_layers_decoder)][1],
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
                    hooks = self.output_hooks_dict[str(self.N_layers_decoder)][1],
                    enable_attention_hooks = enable_attention_hooks,
                    in_chans = in_chans, 
                )
            self.decoder = torch.nn.ModuleDict(module_dict)

        self.module_hooks_dict = {}

    def forward(self, x, input_dict_extra={}):
        # if self.channels_last:
        #     x.contiguous(memory_format=torch.channels_last)
        
        if self.if_share_encoder_over_modalities:
            x_out_encoder, (layer_1, layer_2) = input_dict_extra['shared_encoder_outputs']
        else:
            x_out_encoder, (layer_1, layer_2) = forward_vit(self.opt, self.cfg_ViT, self.encoder, x)
    
        # print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape) # torch.Size([16, 301, 768]) torch.Size([16, 301, 768]) torch.Size([16, 301, 768]) torch.Size([16, 301, 768])

        if self.cfg_ViT.if_share_decoder_over_heads:
            # x = self.decoder()
            x_out_decoder, (layer_3, layer_4) = forward_vit_ViT(self.opt, self.cfg_ViT, self.decoder, x_out_encoder)
            x = x_out_decoder
            x = x.mean(dim = 1) if self.ViT_pool == 'mean' else x[:, 0]
            return x, (layer_1, layer_2, layer_3, layer_4)
        else:
            return_dict_x = {}
            return_dict_layers = {}
            for head_name in self.head_names:
                x_out_decoder, (layer_3, layer_4) = forward_vit_ViT(self.opt, self.cfg_ViT, self.decoder[head_name], x_out_encoder)
                x = x_out_decoder
                x = x.mean(dim = 1) if self.ViT_pool == 'mean' else x[:, 0]
                return_dict_x[head_name] = x
                return_dict_layers[head_name] = (layer_1, layer_2, layer_3, layer_4)
            return return_dict_x, return_dict_layers


def get_ModelAll_ViT(opt, backbone, N_layers_encoder, N_layers_decoder, modalities=[]):
    assert all([x in ['lo', 'ob'] for x in modalities])
    head_names_dict = {
        'al': ['albedo'], 
        'no': ['normal'], 
        'de': ['depth'], 
        'ro': ['rough'], 
        'lo': ['camera', 'layout'], 
        'li': ['lighting']
        }

    module_dict = {}
    for modality in modalities:
        if modality in ['lo']:
            module_dict[modality] = ViTLayoutModel(
                opt=opt, 
                cfg_ViT=opt.cfg.MODEL_LAYOUT_EMITTER.layout.ViT_baseline, 
                modality=modality, 
                backbone=backbone, 
                if_imagenet_backbone=opt.cfg.MODEL_ALL.ViT_baseline.if_imagenet_backbone, 
                if_share_encoder_over_modalities=opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities, 
                N_layers_encoder=N_layers_encoder, 
                N_layers_decoder=N_layers_decoder, 
                head_names=head_names_dict[modality], 
                ViT_pool=opt.cfg.MODEL_ALL.ViT_baseline.ViT_pool
            )
        if modality in ['al', 'no', 'de', 'ro']:
            module_dict[modality] = DPTBRDFModel(
                opt=opt, 
                cfg_DPT=opt.cfg.MODEL_BRDF.DPT_baseline, 
                modality=modality, 
                backbone=backbone, 
                if_imagenet_backbone=opt.cfg.MODEL_BRDF.DPT_baseline.if_imagenet_backbone, 
                if_share_encoder_over_modalities=opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities, 
                non_negative=True if opt.cfg.MODEL_BRDF.DPT_baseline.modality in ['de'] else False,
                enable_attention_hooks=opt.cfg.MODEL_BRDF.DPT_baseline.if_enable_attention_hooks,
                readout=opt.cfg.MODEL_BRDF.DPT_baseline.readout, 
            )

    if opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities:
        module_dict['shared_encoder'] = module_dict[modalities[0]].encoder
        for modality in modalities:
            module_dict[modality].encoder = nn.Identity()

    full_model = torch.nn.ModuleDict(module_dict)
    
    return full_model

class ViTLayoutModel(Transformer_Hybrid_Encoder_Decoder):
    def __init__(
        self, 
        opt, 
        cfg_ViT, 
        modality='lo', 
        N_layers_encoder=6, 
        N_layers_decoder=6, 
        ViT_pool='cls', 
        head_names=[], 
        skip_keys=[], keep_keys=[], **kwargs
    ):
        self.modality = modality
        assert modality in ['lo', 'ob'], 'Invalid modality: %s'%modality
        assert ViT_pool in ['mean', 'cls']

        super().__init__(opt, cfg_ViT=cfg_ViT, head_names=head_names, ViT_pool=ViT_pool, N_layers_encoder=N_layers_encoder, N_layers_decoder=N_layers_decoder, **kwargs)

        if opt.cfg.MODEL_ALL.ViT_baseline.if_indept_MLP_heads:
            self.heads = decoder_layout_emitter_heads_indeptMLP(opt, if_layout=True, 
            if_two_decoders=not cfg_ViT.if_share_decoder_over_heads, 
            if_layer_norm=opt.cfg.MODEL_ALL.ViT_baseline.if_indept_MLP_heads_if_layer_norm)
        else:
            self.heads = decoder_layout_emitter_heads(opt, if_layout=True, if_two_decoders=not cfg_ViT.if_share_decoder_over_heads)

    def forward(self, x, input_dict_extra={}):
        x_out, layers_out = super().forward(x, input_dict_extra=input_dict_extra) # can be tensor + tuple, or dicts of (tensor + tuple)
        if self.modality == 'lo':
            x_out = self.heads(x_out)
        return x_out

class DPTBRDFModel(Transformer_Hybrid_Encoder_Decoder):
    def __init__(
        self, 
        opt, 
        modality='al', 
        path=None, 
        non_negative=False, scale=1.0, shift=0.0, 
        skip_keys=[], keep_keys=[], **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.modality = modality
        assert modality in ['al', 'de', 'ro', 'no'], 'Invalid modality: %s'%modality
        self.out_channels = {'al': 3, 'de': 1, 'ro': 1, 'no': 3}[modality]

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
            assert False
            # print(magenta('===== [DPTBRDFModel] Loading %s'%path))
            # self.load(path, skip_keys=skip_keys, keep_keys=keep_keys)

    def forward(self, x, input_dict_extra={}):
        x_out = super().forward(x, input_dict_extra=input_dict_extra)
        if self.modality == 'al':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
        elif self.modality == 'ro':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
            x_out = torch.mean(x_out, dim=1, keepdim=True)
        elif self.modality == 'de':
            '''
            where x_out is disparity (inversed * baseline)'''
            x_out = self.scale * x_out + self.shift
            x_out[x_out < 1e-8] = 1e-8
            x_out = 1.0 / x_out
        else:
            assert False

        return x_out, {}
