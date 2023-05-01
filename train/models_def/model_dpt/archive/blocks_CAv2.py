import torch
import torch.nn as nn

from .vit_CAv2 import (
    _make_pretrained_vitb_rn50_384_CAv2,
    forward_vit_CAv2,
)
from .blocks import _make_scratch


def _make_encoder_CAv2(
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
    #     scratch = _make_scratch(
    #         [256, 512, 1024, 1024], features, groups=groups, expand=expand
    #     )  # ViT-L/16 - 85.0% Top1 (backbone)
    # elif backbone == "vitb_rn50_384":
    #     pretrained = _make_pretrained_vitb_rn50_384(
    #         use_pretrained,
    #         hooks=hooks,
    #         use_vit_only=use_vit_only,
    #         use_readout=use_readout,
    #         enable_attention_hooks=enable_attention_hooks,
    #     )
    #     scratch = _make_scratch(
    #         [256, 512, 768, 768], features, groups=groups, expand=expand
    #     )  # ViT-H/16 - 85.0% Top1 (backbone)
    if backbone == "vitb_rn50_384_N_layers":
        pretrained = _make_pretrained_vitb_rn50_384_CAv2(
            opt, 
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )  # ViT-H/16
    # elif backbone == "vitb16_384":
    #     pretrained = _make_pretrained_vitb16_384(
    #         use_pretrained,
    #         hooks=hooks,
    #         use_readout=use_readout,
    #         enable_attention_hooks=enable_attention_hooks,
    #     )
    #     scratch = _make_scratch(
    #         [96, 192, 384, 768], features, groups=groups, expand=expand
    #     )  # ViT-B/16 - 84.6% Top1 (backbone)
    # elif backbone == "resnext101_wsl":
    #     pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    #     scratch = _make_scratch(
    #         [256, 512, 1024, 2048], features, groups=groups, expand=expand
    #     )  # efficientnet_lite3
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained, scratch