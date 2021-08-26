import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F
import time
import models_def.models_brdf as models_brdf # basic model

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


def forward_vit_SSN_qkv_yogo(opt, pretrained, x, input_dict_extra={}, hooks=[]):
    b, c, h, w = x.shape

    glob, flex_return_dict = pretrained.model.forward_flex_SSN(opt, x, pretrained.activations, input_dict_extra=input_dict_extra, hooks=hooks)

    if (opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_feat_in_transformer) and not opt.cfg.MODEL_BRDF.DPT_baseline.use_vit_only:
        layer_1 = flex_return_dict['unet_output_dict']['dx4']
        layer_2 = flex_return_dict['unet_output_dict']['dx3']
    else:
        layer_1 = pretrained.activations["1"]
        layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    if_print = True

    # [layer_3 and layer_4 are from transformer layers]
    # print(x.shape)
    if if_print:
        print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # hybrid-SSN: torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768])
    # hybrid-SSN-qkv-unet: torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768])
    # hybrid(-SSN)-ViT: torch.Size([1, 321, 768]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768])

    # print(pretrained.act_postprocess1[0:2])
    # print(pretrained.act_postprocess2[0:2])
    
    # layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    # layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    # layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    # layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    # if if_print:
    #     print('->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # hybrid-SSN: -> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 320]) torch.Size([2, 768, 320])
    # hybrid-SSN-qkv: -> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 320]) torch.Size([1, 768, 320])
    # hybrid-SSN-qkv-unet: -> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 320]) torch.Size([1, 768, 320])
    # hybrid(-SSN)-ViT: -> torch.Size([1, 768, 320]) torch.Size([1, 768, 320]) torch.Size([1, 768, 320]) torch.Size([1, 768, 320])

    assert pretrained.model.patch_size[0]==pretrained.model.patch_size[1]

    # ssn_from = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_from
    # recon_method = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_recon_method
    
    # if recon_method in ['qkv']:
    #     ca_modules = input_dict_extra['ca_modules']
    #     extra_im_scales = 1.
    #     if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone:
    #         extra_im_scales = 0.25
    #     # print(backbone_feat_proj.shape, ssn_return_dict['im_feat'].shape, '++++++')
    #     if layer_1.ndim == 3:
    #         # print(layer_1.shape, ssn_return_dict['im_feat'].shape) # torch.Size([1, 768, 320]) torch.Size([1, 1344, 64, 80])
    #         layer_1 = ca_modules['layer_1_ca'](ssn_return_dict['im_feat'], layer_1, im_feat_scale_factor=extra_im_scales * 1.) # torch.Size([1, 768, 320])
    #     if layer_2.ndim == 3:
    #         layer_2 = ca_modules['layer_2_ca'](ssn_return_dict['im_feat'], layer_2, im_feat_scale_factor=extra_im_scales * 1./2.) # torch.Size([1, 768, 320])
    #     if layer_3.ndim == 3:
    #         # print(layer_3.shape, ssn_return_dict['im_feat'].shape) # torch.Size([1, 768, 320]) torch.Size([1, 1344, 64, 80])
    #         layer_3 = ca_modules['layer_3_ca'](ssn_return_dict['im_feat'], layer_3, im_feat_scale_factor=extra_im_scales * 1./4.) # torch.Size([1, 768, 320])
    #     if layer_4.ndim == 3:
    #         layer_4 = ca_modules['layer_4_ca'](ssn_return_dict['im_feat'], layer_4, im_feat_scale_factor=extra_im_scales * 1./8.) # torch.Size([1, 768, 320])
    # else:
    #     assert False

    # if if_print:
    #     print('-->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # hybrid-SSN: --> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 16, 20]) torch.Size([2, 768, 16, 20])
    # hybrid-SSN-qkv: --> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 16, 20]) torch.Size([1, 768, 8, 10])
    # hybrid-SSN-qkv-unet: --> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 16, 20]) torch.Size([1, 768, 8, 10])

    # print(flex_return_dict['im_feat_dict'].keys())
    layer_1 = flex_return_dict['im_feat_dict']['im_feat_%d'%hooks[0]]
    layer_2 = flex_return_dict['im_feat_dict']['im_feat_%d'%hooks[1]]
    layer_3 = flex_return_dict['im_feat_dict']['im_feat_%d'%hooks[2]]
    layer_4 = flex_return_dict['im_feat_dict']['im_feat_%d'%hooks[3]]

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    if if_print:
        print('---->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # hybrid-SSN: ----> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 16, 20]) torch.Size([2, 768, 8, 10])
    # hybrid-SSN-qkv-unet: ----> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 16, 20]) torch.Size([1, 768, 8, 10])

    return layer_1, layer_2, layer_3, layer_4, flex_return_dict

def forward_flex_SSN_unet_qkv_yogo(self, opt, x, pretrained_activations=[], input_dict_extra={}, hooks=[]):
    b, c, im_height, im_width = x.shape # image pixel space

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        output_resnet = self.patch_embed.backbone(x) # [patch_embed] https://github.com/rwightman/pytorch-image-models/blob/72b227dcf57c0c62291673b96bdc06576bb90457/timm/models/layers/patch_embed.py#L15

    assert all([x!=1 for x in self.patch_size])
    assert im_height%self.patch_size[0]==0 and im_width%self.patch_size[1]==0

    im_feat_init = torch.cat(
        [
            F.interpolate(pretrained_activations['feat_stem'], scale_factor=1, mode='bilinear'), # torch.Size([4, 64, 64, 80])
            F.interpolate(pretrained_activations['feat_stage_0'], scale_factor=1, mode='bilinear'), # torch.Size([4, 256, 64, 80])
            F.interpolate(pretrained_activations['feat_stage_1'], scale_factor=2, mode='bilinear'), # torch.Size([4, 512, 32, 40])
            F.interpolate(pretrained_activations['feat_stage_2'], scale_factor=2, mode='bilinear'), # torch.Size([4, 512, 32, 40])
            F.interpolate(output_resnet, scale_factor=4, mode='bilinear'), # torch.Size([4, 1024, 16, 20])
        ], dim=1)
    im_feat_init = self.patch_embed.proj_extra(im_feat_init)
    
    batch_size, d = im_feat_init.shape[0], im_feat_init.shape[1]
    spixel_dims = [im_height//self.patch_size[0], im_width//self.patch_size[1]]

    ssn_op = SSNFeatsTransformAdaptive(None, spixel_dims=spixel_dims, if_dense=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_dense)

    mask_resized = input_dict_extra['brdf_loss_mask']

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_from == 'matseg':
        ssn_return_dict = ssn_op(tensor_to_transform=im_feat_init, feats_in=input_dict_extra['return_dict_matseg']['embedding'], mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1, 1./4.)) # Q: [im_height, im_width]
    else:
        ssn_return_dict = ssn_op(tensor_to_transform=im_feat_init, feats_in=pretrained_activations['feat_stage_2'], mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1./2., 1)) # Q: [im_height/4, im_width/4]

    abs_affinity = ssn_return_dict['abs_affinity'] # fixed for now
    
    c = ssn_return_dict['C'] # codebook
    c = c.view([batch_size, d, spixel_dims[0], spixel_dims[1]])

    # x = self.patch_embed.proj(c).flatten(2).transpose(1, 2) # torch.Size([8, 320, 768]); will be Identity op in qkv recon
    x = c.flatten(2).transpose(1, 2) # torch.Size([8, 320, 768]); will be Identity op in qkv recon


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
    # x = self.pos_drop(x)
    ca_modules = input_dict_extra['ca_modules']

    im_feat_dict = {'im_feat_-1': im_feat_init}
    extra_im_scales = [1., 1./2., 1./2., 1./2.]

    extra_im_scale_accu = 1.

    for idx, blk in enumerate(self.blocks):
        x = blk(x) # [-1, 768, 321]
        x_cls_token = x[:, 0:1, :] # [-1, 768, 1]
        x_tokens = x[:, 1:, :].transpose(1, 2) # [-1, 768, 320]

        if idx in hooks:
            idx_within_hooks = hooks.index(idx)
            extra_im_scale = extra_im_scales[idx_within_hooks]
            extra_im_scale_accu *= extra_im_scale
        else:
            extra_im_scale = 1.


        im_feat_idx = ca_modules['layer_%d_ca'%idx](im_feat_dict['im_feat_%d'%(idx-1)], x_tokens, im_feat_scale_factor=extra_im_scale) # torch.Size([1, 768, 320])
        # print(idx, extra_im_scale_accu, im_feat_dict['im_feat_%d'%(idx-1)].shape, extra_im_scale, im_feat_idx.shape)
        im_feat_dict['im_feat_%d'%idx] = im_feat_idx

        assert opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_from == 'matseg', 'only supporting this now'
        # print(im_feat_idx.shape, abs_affinity.shape, mask_resized.shape) # torch.Size([1, 768, 64, 80]) torch.Size([1, 320, 256, 320]) torch.Size([1, 256, 320])
        ssn_return_dict_idx = ssn_op(tensor_to_transform=im_feat_idx, affinity_in=abs_affinity, mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1, 1./4.*extra_im_scale_accu)) # Q: [im_height, im_width]
        c_idx = ssn_return_dict_idx['C']
        x_prime = c_idx.flatten(2).transpose(1, 2) # torch.Size([8, 320, 768]); will be Identity op in qkv recon
        x_prime = torch.cat((x_cls_token, x_prime), dim=1)
        x = x_prime

        # print(c_idx.shape, x_prime.shape) # torch.Size([1, 768, 320]) torch.Size([1, 320, 768])

    x = self.norm(x)

    extra_return_dict = {'Q': ssn_return_dict['Q'], 'matseg_affinity': ssn_return_dict['Q_2D'], 'im_feat': im_feat_init, 'im_feat_dict': im_feat_dict}

    return x, extra_return_dict