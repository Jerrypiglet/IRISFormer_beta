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

class Transformer_Hybrid_Encoder_Decoder(BaseModel):
    '''
    ViT for one modality; can have multiple heads
    '''
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
    ):

        super(Transformer_Hybrid_Encoder_Decoder, self).__init__()

        self.channels_last = channels_last
        self.cfg_ViT = cfg_ViT

        self.opt = opt
        self.N_layers_encoder = N_layers_encoder
        self.N_layers_decoder = N_layers_decoder
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

        self.if_share_decoder_over_heads = len(self.head_names) <= 1 and cfg_ViT.if_share_decoder_over_heads
        if self.if_share_decoder_over_heads:
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
            x_out_encoder, (layer_1, layer_2) = forward_vit_ViT_encoder(self.opt, self.cfg_ViT, self.encoder, x)
    
        # print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape) # torch.Size([16, 301, 768]) torch.Size([16, 301, 768]) torch.Size([16, 301, 768]) torch.Size([16, 301, 768])
    
        if self.if_share_decoder_over_heads:
            if self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities \
                 and len(self.head_names) == 1 and self.head_names[0] in ['al', 'ro', 'de', 'no']:
                    x_out_decoder, (layer_3, layer_4) = input_dict_extra['shared_BRDF_decoder_outputs']
            else:        
                x_out_decoder, (layer_3, layer_4) = forward_vit_ViT_decoder(self.opt, self.cfg_ViT, self.decoder, x_out_encoder)
            return x_out_decoder, [layer_1, layer_2, layer_3, layer_4]
        else:
            return_dict_x = {}
            return_dict_layers = {}
            for head_name in self.head_names:
                x_out_decoder, (layer_3, layer_4) = forward_vit_ViT_decoder(self.opt, self.cfg_ViT, self.decoder[head_name], x_out_encoder)
                return_dict_x[head_name] = x_out_decoder
                return_dict_layers[head_name] = [layer_1, layer_2, layer_3, layer_4]
            return return_dict_x, return_dict_layers