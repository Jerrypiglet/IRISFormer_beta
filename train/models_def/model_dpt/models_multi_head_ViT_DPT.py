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
    forward_vit_ViT_encoder,
    forward_vit_ViT_decoder,
    _make_pretrained
)
from .blocks import (
    Interpolate,
)
from .vit_ViT import (
    forward_flex_decoder
)

from .models import _make_fusion_block

from .models_ViT import decoder_layout_emitter_heads_indeptMLP, decoder_layout_emitter_heads

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

class ModelAll_ViT(torch.nn.Module):
    '''
    ViT/DPT for multiple modalities
    '''
    def __init__(
        self, opt, backbone, N_layers_encoder, N_layers_decoder, modalities=[]):

        super(ModelAll_ViT, self).__init__()

        self.opt = opt

        head_names_dict = {
            'al': ['albedo'], 
            'no': ['normal'], 
            'de': ['depth'], 
            'ro': ['rough'], 
            'lo': ['camera', 'layout'], 
            'li': ['lighting']
            }
        assert all([x in list(head_names_dict.keys()) for x in modalities])

        module_dict = {}
        for modality in modalities:
            if modality in ['lo']:
                module_dict[modality] = ViTLayoutObjModel(
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
                    features=256,
                    groups=1,
                    expand=False,
                    if_imagenet_backbone=opt.cfg.MODEL_BRDF.DPT_baseline.if_imagenet_backbone, 
                    if_share_encoder_over_modalities=opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities, 
                    N_layers_encoder=N_layers_encoder, 
                    N_layers_decoder=N_layers_decoder, 
                    non_negative=True if modality in ['de'] else False,
                    # enable_attention_hooks=opt.cfg.MODEL_BRDF.DPT_baseline.if_enable_attention_hooks,
                    DPT_readout=opt.cfg.MODEL_BRDF.DPT_baseline.readout, 
                )

        if opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities:
            module_dict['shared_encoder'] = module_dict[modalities[0]].encoder
            for modality in modalities:
                module_dict[modality].encoder = nn.Identity()

        if any([x in ['al', 'no', 'de', 'ro'] for x in modalities]):
            modalities_BRDF = list(set(['al', 'no', 'de', 'ro']) & set(modalities))
            if opt.cfg.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities:
                module_dict['shared_BRDF_decoder'] = module_dict[modalities_BRDF[0]].decoder
                for modality in modalities_BRDF:
                    module_dict[modality].decoder = nn.Identity()

        self._ = torch.nn.ModuleDict(module_dict)
        
    def forward(self, x):
        input_dict_extra = {}

        assert self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities
        input_dict_extra['shared_encoder_outputs'] = forward_vit_ViT_encoder(
            self.opt, self.opt.cfg.MODEL_ALL.ViT_baseline, self._.shared_encoder, 
            x)

        if self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities:
            input_dict_extra['shared_BRDF_decoder_outputs'] = forward_vit_ViT_decoder(
                self.opt, self.opt.cfg.MODEL_ALL.ViT_baseline, self._.shared_BRDF_decoder, 
                input_dict_extra['shared_encoder_outputs'][0])

        output_dict = {}
        modalities = self.opt.cfg.MODEL_ALL.enable_list
        for modality in modalities:
            output_dict[modality] = self._[modality].forward(None, input_dict_extra=input_dict_extra)

        return output_dict


class ViTLayoutObjModel(Transformer_Hybrid_Encoder_Decoder):
    def __init__(
        self, 
        opt, 
        cfg_ViT, 
        modality='lo', 
        N_layers_encoder=6, 
        N_layers_decoder=6, 
        ViT_pool = 'mean',  # ViT_pool strategy in the end: 'cls' or 'mean'
        head_names=[], 
        skip_keys=[], keep_keys=[], **kwargs
    ):  
        self.cfg_ViT = cfg_ViT
        self.modality = modality
        assert modality in ['lo', 'ob'], 'Invalid modality: %s'%modality

        self.ViT_pool = ViT_pool
        assert ViT_pool in ['mean', 'cls']

        super().__init__(opt, cfg_ViT=cfg_ViT, head_names=head_names, N_layers_encoder=N_layers_encoder, N_layers_decoder=N_layers_decoder, **kwargs)

        if opt.cfg.MODEL_ALL.ViT_baseline.if_indept_MLP_heads:
            self.heads = decoder_layout_emitter_heads_indeptMLP(opt, if_layout=True, 
            if_two_decoders=not cfg_ViT.if_share_decoder_over_heads, 
            if_layer_norm=opt.cfg.MODEL_ALL.ViT_baseline.if_indept_MLP_heads_if_layer_norm)
        else:
            self.heads = decoder_layout_emitter_heads(opt, if_layout=True, if_two_decoders=not cfg_ViT.if_share_decoder_over_heads)

    def forward(self, x, input_dict_extra={}):
        decoder_out, layers_out = super().forward(x, input_dict_extra=input_dict_extra) # can be tensor + tuple, or dicts of (tensor + tuple)

        if self.cfg_ViT.if_share_decoder_over_heads:
            x_out = decoder_out.mean(dim = 1) if self.ViT_pool == 'mean' else decoder_out[:, 0]
        else:
            x_out = {}
            for head_name in ['camera', 'layout']:
                x_out[head_name] = decoder_out[head_name].mean(dim = 1) if self.ViT_pool == 'mean' else decoder_out[head_name][:, 0]

        if self.modality == 'lo':
            x_out = self.heads(x_out)
        return x_out

class DPTBRDFModel(Transformer_Hybrid_Encoder_Decoder):
    def __init__(
        self, 
        opt, 
        cfg_DPT, 
        modality='al', 
        N_layers_encoder=6, 
        N_layers_decoder=6, 
        if_upscale_last_layer=True, 
        groups=1,
        expand=False,
        use_bn=False,
        DPT_readout = 'ignore', 
        non_negative=False, scale=1.0, shift=0.0, 
        skip_keys=[], keep_keys=[], **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.modality = modality
        assert modality in ['al', 'de', 'ro', 'no'], 'Invalid modality: %s'%modality
        self.out_channels = {'al': 3, 'de': 1, 'ro': 1, 'no': 3}[modality]
        self.if_batch_norm = opt.cfg.MODEL_BRDF.DPT_baseline.if_batch_norm

        self.scale = scale
        self.shift = shift


        super().__init__(opt, cfg_ViT=cfg_DPT, head_names=[modality], N_layers_encoder=N_layers_encoder, N_layers_decoder=N_layers_decoder, **kwargs)

        self.scratch = _make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )  # ViT-H/16
        self.scratch.refinenet1 = _make_fusion_block(opt, features, use_bn, if_upscale=if_upscale_last_layer)
        self.scratch.refinenet2 = _make_fusion_block(opt, features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(opt, features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(opt, features, use_bn)
        output_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32) if self.if_batch_norm else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(32, self.out_channels, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Tanh() if non_negative else nn.Identity(),

        )
        self.scratch.output_conv = output_head

        self.pretrained = _make_pretrained(
            size=[opt.cfg.DATA.im_height_padded, opt.cfg.DATA.im_width_padded],
            features=[256, 512, 768, 768],
            use_readout=DPT_readout,
        )
        self.unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        opt.cfg.DATA.im_height_padded // cfg_DPT.patch_size,
                        opt.cfg.DATA.im_width_padded // cfg_DPT.patch_size,
                    ]
                ),
            )
        )

    

    def forward(self, x, input_dict_extra={}):
        decoder_out, layers_out = super().forward(x, input_dict_extra=input_dict_extra)
        
        layer_1, layer_2, layer_3, layer_4 = layers_out[0], layers_out[1], layers_out[2], layers_out[3]

        layer_1 = self.pretrained.act_postprocess1[0:2](layer_1)
        layer_2 = self.pretrained.act_postprocess2[0:2](layer_2)
        layer_3 = self.pretrained.act_postprocess3[0:2](layer_3)
        layer_4 = self.pretrained.act_postprocess4[0:2](layer_4)
        # print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
        if layer_1.ndim == 3:
            assert False, 'should be ResNet feats in DPT-hybrid setting!'
            layer_1 = self.unflatten(layer_1)
        if layer_2.ndim == 3:
            assert False, 'should be ResNet feats in DPT-hybrid setting!'
            layer_2 = self.unflatten(layer_2)
        if layer_3.ndim == 3:
            layer_3 = self.unflatten(layer_3)
        if layer_4.ndim == 3:
            layer_4 = self.unflatten(layer_4)

        layer_1 = self.pretrained.act_postprocess1[3:](layer_1)
        layer_2 = self.pretrained.act_postprocess2[3:](layer_2)
        layer_3 = self.pretrained.act_postprocess3[3:](layer_3)
        layer_4 = self.pretrained.act_postprocess4[3:](layer_4)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        x_out = self.scratch.output_conv(path_1)

        if self.modality == 'al':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
        elif self.modality == 'no':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
            norm = torch.sqrt(torch.sum(x_out * x_out, dim=1).unsqueeze(1) ).expand_as(x_out)
            x_out = x_out / torch.clamp(norm, min=1e-6)
        elif self.modality == 'ro':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
            x_out = torch.mean(x_out, dim=1, keepdim=True)
        elif self.modality == 'de':
            '''
            where x_out is disparity (inversed * baseline)'''
            print(torch.max(x_out), torch.min(x_out), torch.median(x_out))
            x_out = 0.5 * (x_out + 1) # [-1, 1] -> [0, 1]
            x_out = self.scale * x_out + self.shift
            x_out[x_out < 1e-8] = 1e-8
            x_out = 1.0 / x_out
        else:
            assert False

        return x_out
