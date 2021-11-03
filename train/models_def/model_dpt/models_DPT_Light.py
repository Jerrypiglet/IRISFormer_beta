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

class DPTLightModel(Transformer_Hybrid_Encoder_Decoder):
    def __init__(
        self, 
        opt, 
        cfg_DPT, 
        modality, 
        SGNum,
        N_layers_encoder=6, 
        N_layers_decoder=6, 
        if_upscale_last_layer=False, 
        groups=1,
        expand=False,
        use_bn=False,
        DPT_readout = 'ignore', 
        skip_keys=[], keep_keys=[], **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.SGNum = SGNum

        self.modality = modality
        assert modality in ['axis', 'lamb', 'weight'], 'Invalid modality: %s'%modality
        self.out_channels = {'axis': 3*SGNum, 'lamb': SGNum, 'weight': 3*SGNum}[modality]

        self.if_batch_norm = opt.cfg.MODEL_LIGHT.DPT_baseline.if_batch_norm
        # self.if_group_norm = opt.cfg.MODEL_LIGHT.DPT_baseline.if_group_norm
        if modality == 'weight':
            self.if_batch_norm = opt.cfg.MODEL_LIGHT.DPT_baseline.if_batch_norm_weight_override

        super().__init__(opt, cfg_ViT=cfg_DPT, head_names=[modality], N_layers_encoder=N_layers_encoder, N_layers_decoder=N_layers_decoder, **kwargs)

        self.scratch = _make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )  # ViT-H/16
        self.scratch.refinenet1 = _make_fusion_block(opt, features, use_bn, if_upscale=if_upscale_last_layer)
        self.scratch.refinenet2 = _make_fusion_block(opt, features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(opt, features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(opt, features, use_bn)

        output_head = nn.Sequential(
                nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(features//2) if self.if_batch_norm else nn.Identity(),
                nn.ReLU(True),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(features//2, 64, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(64) if self.if_batch_norm else nn.Identity(),
                nn.Identity(), 
                nn.ReLU(True),
                nn.Conv2d(64, self.out_channels, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm2d(self.out_channels) if self.if_batch_norm else nn.Identity(),
                # nn.GroupNorm(self.out_channels) if self.if_group_norm else nn.Identity(),
                # nn.ReLU(True) if non_negative else nn.Identity(),
            )

        self.scratch.output_conv = output_head

        self.pretrained = _make_pretrained(
            cfg_DPT=cfg_DPT, 
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

        if self.modality in ['lamb']:
            x_out = torch.tanh(x_out )
            x_out = 0.5 * (x_out + 1)
            x_out = torch.clamp(x_out, min=0.)
        elif self.modality in ['weight']:
            # x_out = self.relu(x_out )
            x_out = torch.tanh(x_out )
            x_out = 0.5 * (x_out + 1)
            x_out = torch.clamp(x_out, min=0.)
        elif self.modality in ['axis']:
            bn, _, row, col = x_out.size()
            x_out = x_out.view(bn, self.SGNum, 3, row, col)
            x_out = x_out / torch.clamp(torch.sqrt(torch.sum(x_out * x_out,
                dim=2).unsqueeze(2) ), min = 1e-6).expand_as(x_out )
        else:
            assert False
        return x_out