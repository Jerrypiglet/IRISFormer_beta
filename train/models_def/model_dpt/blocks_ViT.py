import torch
import torch.nn as nn

from .vit_ViT import (
    _make_pretrained_vitb_rn50_384_ViT,
    # _make_pretrained_vitl16_384,
    # _make_pretrained_vitb16_384,
    forward_vit_ViT,
)


def _make_encoder_ViT(
    cfg_DPT, 
    backbone,
    use_pretrained,
    num_layers, 
    hooks=None,
    enable_attention_hooks=False,
    in_chans=3
):
    if backbone in ["vitb_rn50_384", "vitb_rn50_384_N_layers"]:
        pretrained = _make_pretrained_vitb_rn50_384_ViT(
            cfg_DPT, 
            use_pretrained,
            num_layers=num_layers, 
            hooks=hooks,
            enable_attention_hooks=enable_attention_hooks,
            in_chans=in_chans
        )
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained

def _make_decoder_ViT(
    cfg_DPT, 
    backbone,
    use_pretrained,
    num_layers, 
    hooks=None,
    enable_attention_hooks=False,
    in_chans=3,
):
    if backbone in ["vitb_rn50_384", "vitb_rn50_384_N_layers"]:
        pretrained = _make_pretrained_vitb_rn50_384_ViT(
            cfg_DPT, 
            use_pretrained,
            num_layers=num_layers, 
            hooks=hooks,
            enable_attention_hooks=enable_attention_hooks,
            in_chans=in_chans, 
            if_transformer_only=True
        )
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained

import torch.nn.functional as F

# ViT-pooling from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# FFN heads from DTER (https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/detr.py#L289)
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x