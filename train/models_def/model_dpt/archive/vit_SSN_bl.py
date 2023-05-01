import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F
import time

# from models_def.model_matseg import Baseline

from models_def.model_nvidia.AppGMM_adaptive import SSNFeatsTransformAdaptive

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

attention = {}

def get_attention(name):
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

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook

from .vit import (
    Transpose, 
    get_readout_oper
)

def QtC(codebook, gamma, Q_height, Q_width, Q_downsample_rate=1):
    '''
    C: [B, D, J]
    gamma: [B, J, N]
    '''
    assert len(gamma.shape)==3
    batch_size, J, N = gamma.shape
    D = codebook.shape[1]
    assert Q_height * Q_width == N

    if Q_downsample_rate != 1:
        gamma_resampled = gamma.view(batch_size, J, Q_height, Q_width)
        gamma_resampled = F.interpolate(gamma_resampled, scale_factor=1./float(Q_downsample_rate), mode='bilinear')
        gamma_resampled = gamma_resampled / (torch.sum(gamma_resampled, 1, keepdims=True)+1e-6)
        gamma_resampled = gamma_resampled.view(batch_size, J, -1)
    else:
        gamma_resampled = gamma

    # print(Q_downsample_rate, codebook.shape, gamma.shape, gamma_resampled.shape) # 16 torch.Size([2, 768, 320]) torch.Size([2, 320, 320])
    im_hat = codebook @ gamma_resampled
    im_hat = im_hat.view(batch_size, D, Q_height//Q_downsample_rate, Q_width//Q_downsample_rate)
    return im_hat

def _make_pretrained_vitb16_384_SSN(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    print('========= [_make_pretrained_vitb16_384_SSN] pretrained', pretrained)

    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained)
    model.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=1, stride=1)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    recon_method = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_recon_method

    # 32, 48, 136, 384
    act_postprocess1_list = [
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    ]
    if recon_method in ['qkv', 'qtc']:
        act_postprocess1_list.pop()
    pretrained.act_postprocess1 = nn.Sequential(*act_postprocess1_list)

    
    act_postprocess2_list = [
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    ]
    if recon_method in ['qkv', 'qtc']:
        act_postprocess2_list.pop()
    pretrained.act_postprocess2 = nn.Sequential(*act_postprocess2_list)

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

    act_postprocess4_list = [
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
    ]
    if recon_method in ['qkv', 'qtc']:
        act_postprocess4_list.pop()
    pretrained.act_postprocess4 = nn.Sequential(*act_postprocess4_list)

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex_SSN = types.MethodType(forward_flex_SSN_bl, pretrained.model)
    # pretrained.model._resize_pos_embed = types.MethodType(
    #     _resize_pos_embed, pretrained.model
    # )

    return pretrained


def forward_flex_SSN_bl(self, opt, x, pretrained_activations=[], input_dict_extra={}):
    b, c, im_height, im_width = x.shape # image pixel space

    # pos_embed = self._resize_pos_embed_SSN_b(
    #     self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    # )

    B = x.shape[0]

    assert all([x!=1 for x in self.patch_size])
    assert im_height%self.patch_size[0]==0 and im_width%self.patch_size[1]==0

    im_feat = x
    
    batch_size, d = im_feat.shape[0], im_feat.shape[1]
    spixel_dims = [im_height//self.patch_size[0], im_width//self.patch_size[1]]

    ssn_op = SSNFeatsTransformAdaptive(None, spixel_dims=spixel_dims)

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.ssn_from == 'matseg':
        ssn_return_dict = ssn_op(tensor_to_transform=im_feat, feats_in=input_dict_extra['return_dict_matseg']['embedding'], if_return_codebook_only=True, scale_down_gamma_tensor=(1, 1)) # Q: [im_height, im_width]
    else:
        ssn_return_dict = ssn_op(tensor_to_transform=im_feat, feats_in=im_feat, if_return_codebook_only=True, scale_down_gamma_tensor=(1, 1)) # Q: [im_height, im_width]
    
    c = ssn_return_dict['C'] # codebook
    c = c.view([batch_size, d, spixel_dims[0], spixel_dims[1]])

    # print(ssn_return_dict['C'].shape, ssn_return_dict['Q_2D'].shape) # torch.Size([2, 1344, 320]) torch.Size([2, 320, 256, 320])
    x = self.patch_embed.proj(c).flatten(2).transpose(1, 2) # torch.Size([8, 320, 768])

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

    # x = x + pos_embed
    # print(x.shape, pos_embed.shape) # torch.Size([8, 321, 768]) torch.Size([1, 321, 768])
    x = self.pos_drop(x)
    for idx, blk in enumerate(self.blocks):
        x = blk(x) # always [8, 321, 768]

    x = self.norm(x)

    extra_return_dict = {'Q': ssn_return_dict['Q'], 'matseg_affinity': ssn_return_dict['Q_2D']}

    return x, extra_return_dict