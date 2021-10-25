import torch
import torch.nn as nn

from .blocks import (
    _make_scratch, 
)
from .models import _make_fusion_block
from .blocks import (
    Interpolate,
)
from .blocks_ViT import (
    _make_pretrained
)

from .models_base_ViT_DPT import Transformer_Hybrid_Encoder_Decoder


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
        if modality == 'de':
            self.if_batch_norm = opt.cfg.MODEL_BRDF.DPT_baseline.if_batch_norm_depth_override

        self.scale = scale
        self.shift = shift
        self.non_negative = non_negative

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
            nn.Tanh() if self.non_negative else nn.Identity(),

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

        if self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_pretrained_over_BRDF_modalities:
            assert 'shared_BRDF_pretrained_outputs' in input_dict_extra
            layer_1, layer_2, layer_3, layer_4 = input_dict_extra['shared_BRDF_pretrained_outputs']
        else:
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
            print(torch.max(x_out), torch.min(x_out), torch.median(x_out), self.non_negative, self.if_batch_norm)
            x_out = 0.5 * (x_out + 1) # [-1, 1] -> [0, 1]
            x_out = self.scale * x_out + self.shift
            # x_out[x_out < 1e-8] = 1e-8
            x_out = 1.0 / (x_out + 1e-8)
        else:
            assert False

        return x_out
