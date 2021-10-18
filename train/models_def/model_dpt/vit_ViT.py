import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}


def get_attention(name):
    '''
    relative pos embed: https://github.com/rwightman/pytorch-image-models/blob/072155951104230c2b5f3bbfb31acc694ee2fa0a/timm/models/layers/bottleneck_attn.py#L55'''
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # print(q.shape, module.num_heads, k.shape, v.shape) # torch.Size([2, 12, 321, 64]) 12 torch.Size([2, 12, 321, 64]) torch.Size([2, 12, 321, 64])  

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

        # print(module.)

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1).contiguous()
        return x


def forward_vit_ViT(opt, cfg_DPT, pretrained, x, input_dict_extra={}):
    b, c, h, w = x.shape

    glob, extra_output_dict = pretrained.model.forward_flex(opt, cfg_DPT, x, input_dict_extra=input_dict_extra)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]
    
    return layer_1, layer_2, layer_3, layer_4


def _resize_pos_embed(self, posemb, gs_h, gs_w): # original at https://github.com/rwightman/pytorch-image-models/blob/72b227dcf57c0c62291673b96bdc06576bb90457/timm/models/vision_transformer.py#L481
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, opt, cfg_DPT, x, input_dict_extra={}):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        # if cfg_DPT.if_share_patchembed:
        #     x = input_dict_extra['shared_patch_embed_backbone_output']
        # else:
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features; 1/16

    # print(x.shape) # torch.Size([1, 1024, 16, 20])
    im_feat_init = self.patch_embed.proj(x)
    x = im_feat_init.flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    if cfg_DPT.if_pos_embed:
        x = x + pos_embed
    x = self.pos_drop(x)

    for idx, blk in enumerate(self.blocks):
    #     if cfg_DPT.dpt_hybrid.N_layers!=-1 and idx >= cfg_DPT.dpt_hybrid.N_layers:
    #         continue
        x = blk(x)
        
    x = self.norm(x)

    return x, {}


def forward_flex_decoder(self, opt, cfg_DPT, x, input_dict_extra={}):

    B = x.shape[0]

    for idx, blk in enumerate(self.blocks):
        x = blk(x)

    return x, {}


def _make_vit_b_rn50_backbone_ViT(
    cfg_DPT, 
    model,
    features=[256, 512, 768, 768],
    size=[384, 384],
    hooks=[0, 1, 8, 11],
    vit_features=768,
    start_index=1,
    enable_attention_hooks=False,
    if_transformer_only=False
):
    pretrained = nn.Module()

    pretrained.model = model

    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    if enable_attention_hooks:
        pretrained.model.blocks[2].attn.register_forward_hook(get_attention("attn_1"))
        pretrained.model.blocks[5].attn.register_forward_hook(get_attention("attn_2"))
        pretrained.model.blocks[8].attn.register_forward_hook(get_attention("attn_3"))
        pretrained.model.blocks[11].attn.register_forward_hook(get_attention("attn_4"))
        pretrained.attention = attention

    pretrained.activations = activations

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [cfg_DPT.patch_size]*2
    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    if if_transformer_only:
        pretrained.model.forward_flex = types.MethodType(forward_flex_decoder, pretrained.model)
    else:
        pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitb_rn50_384_ViT(
    cfg_DPT, 
    pretrained,
    hooks=None,
    num_layers=8, 
    enable_attention_hooks=False,
    in_chans=3, 
    if_transformer_only=False
):
    print('========= [_make_pretrained_vitb_rn50_384] pretrained', pretrained, if_transformer_only) # /home/ruizhu/anaconda3/envs/py38/lib/python3.8/site-packages/timm/models/vision_transformer.py, L570

    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained, in_chans=in_chans)

    if num_layers < 8:
        for layer_idx in range(len(model.blocks)):
            if layer_idx >= num_layers:
                model.blocks[layer_idx] = nn.Identity()

    model_re = _make_vit_b_rn50_backbone_ViT(
        cfg_DPT, 
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        enable_attention_hooks=enable_attention_hooks,
        if_transformer_only=if_transformer_only
    )
    
    if if_transformer_only:
        # model_re.model.cls_token = nn.Identity()
        # model_re.model.pos_embed = nn.Identity()
        # model_re.model.patch_embed = nn.Identity()
        model_rere = nn.Module()
        model_rere.model = nn.Module()
        model_rere.model.blocks = model_re.model.blocks
        return model_rere
    else:
        return model_re
