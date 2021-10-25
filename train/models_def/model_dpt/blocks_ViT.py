import torch
import torch.nn as nn

from .vit_ViT import (
    _make_pretrained_vitb_rn50_384_ViT,
    # _make_pretrained_vitl16_384,
    # _make_pretrained_vitb16_384,
    forward_vit_ViT_encoder,
    forward_vit_ViT_decoder,
    forward_DPT_pretrained
)
from .vit import (
    get_readout_oper, 
    Transpose
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
            if_decoder=True
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

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, if_layer_norm=False):
        super().__init__()
        self.num_layers = num_layers
        self.if_layer_norm = if_layer_norm
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if self.if_layer_norm:
            self.layer_norms = nn.ModuleList(nn.LayerNorm(k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.if_layer_norm and i < self.num_layers - 1:
                x = self.layer_norms[i](x)
        return x

def _make_pretrained(
        size=[384, 384],
        features=[256, 512, 768, 768],
        vit_features=768,
        use_readout="ignore",
        start_index=1,
    ):
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    pretrained = nn.Module()
    pretrained.act_postprocess1 = nn.Sequential(
        nn.Identity(), nn.Identity(), nn.Identity()
    )
    pretrained.act_postprocess2 = nn.Sequential(
        nn.Identity(), nn.Identity(), nn.Identity()
    )
    
    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    return pretrained
