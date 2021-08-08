import torch
import torch.nn as nn
import torch.nn.functional as F
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
)

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
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
        assert backbone in ['vitb_unet_384'], 'Backbone %s unsupported in DPT_SSN!'%backbone

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb_unet_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder_SSN(
            opt, 
            backbone,
            features,
            use_pretrained=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.use_pretrained_backbone,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_vit_only=opt.cfg.MODEL_BRDF.DPT_baseline.use_vit_only, 
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x, input_dict_extra={}):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        # tic = time.time()
        layer_1, layer_2, layer_3, layer_4, ssn_return_dict = forward_vit_SSN(self.opt, self.pretrained, x, input_dict_extra=input_dict_extra)
        # print(time.time() - tic, '------------ forward_vit_SSN')

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

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

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, self.out_channels, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(opt, head, **kwargs)

        if path is not None:
            self.load(path, skip_keys=skip_keys, keep_keys=keep_keys)

    def forward(self, x, input_dict_extra={}):
        x_out, ssn_return_dict = super().forward(x, input_dict_extra=input_dict_extra)
        if self.modality == 'al':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
        elif self.modality == 'de':
            x_out = torch.clamp(x_out, 1e-8, 100)

        return x_out, ssn_return_dict