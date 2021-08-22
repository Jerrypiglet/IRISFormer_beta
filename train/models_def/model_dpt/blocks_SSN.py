import torch
import torch.nn as nn

from .vit_SSN_rn50 import (
    _make_pretrained_vitb_rn50_384_SSN,
)
from .vit_SSN_unet import (
    _make_pretrained_vitb_unet_384_SSN,
    _make_pretrained_vitl_unet_384_SSN,
    forward_vit_SSN
)
from .vit_SSN_bl import (
    _make_pretrained_vitb16_384_SSN,
)
from .vit import (
    _make_pretrained_vitl16_384,
)


def _make_encoder_SSN(
    opt, 
    backbone,
    features,
    use_pretrained,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="ignore",
    enable_attention_hooks=False,
):

    # if backbone == "vitl16_384":
    #     pretrained = _make_pretrained_vitl16_384(
    #         use_pretrained,
    #         hooks=hooks,
    #         use_readout=use_readout,
    #         enable_attention_hooks=enable_attention_hooks,
    #     )
    #     scratch = _make_scratch_SSN(
    #         [256, 512, 1024, 1024], features, groups=groups, expand=expand
    #     )  # ViT-L/16 - 85.0% Top1 (backbone)
    # elif backbone == "vitb_rn50_384":
    #     pretrained = _make_pretrained_vitb_rn50_384_SSN(
    #         use_pretrained,
    #         hooks=hooks,
    #         use_vit_only=use_vit_only,
    #         use_readout=use_readout,
    #         enable_attention_hooks=enable_attention_hooks,
    #     )
    #     scratch = _make_scratch_SSN(
    #         [256, 512, 768, 768], features, groups=groups, expand=expand
    #     )  # ViT-H/16 - 85.0% Top1 (backbone)

    if backbone == "vitb16_384":
        pretrained = _make_pretrained_vitb16_384_SSN(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch_SSN(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # ViT-B/16 - 84.6% Top1 (backbone)
    # elif backbone == "resnext101_wsl":
    #     pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    #     scratch = _make_scratch_SSN(
    #         [256, 512, 1024, 2048], features, groups=groups, expand=expand
    #     )  # efficientnet_lite3
    elif backbone == "vitb_unet_384": # DPT-hybrid-SSN
        if_unet_feat_in_transformer = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_feat_in_transformer

        pretrained = _make_pretrained_vitb_unet_384_SSN(
            opt, 
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        channels = [256, 512, 768, 768] if not if_unet_feat_in_transformer else [128, 256, 768, 768]
        scratch = _make_scratch_SSN(
            channels, features, groups=groups, expand=expand
        )
    elif backbone == "vitl_unet_384": # DPT-large-SSN
        pretrained = _make_pretrained_vitl_unet_384_SSN(
            opt, 
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch_SSN(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained, scratch


def _make_scratch_SSN(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch
