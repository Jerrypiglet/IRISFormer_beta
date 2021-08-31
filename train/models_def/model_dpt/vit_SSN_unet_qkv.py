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

    if_print = False

    # [layer_3 and layer_4 are from transformer layers]
    # print(x.shape)
    if if_print:
        print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # hybrid-SSN: torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768])
    # hybrid-SSN-qkv-unet: torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768])
    # hybrid(-SSN)-ViT: torch.Size([1, 321, 768]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768])

    assert pretrained.model.patch_size[0]==pretrained.model.patch_size[1]

    # print(flex_return_dict['im_feat_dict'].keys())
    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
        pass
    else:
        layer_1 = flex_return_dict['im_feat_dict']['im_feat_%d'%hooks[0]]
        layer_2 = flex_return_dict['im_feat_dict']['im_feat_%d'%hooks[1]]
        layer_3 = flex_return_dict['im_feat_dict']['im_feat_%d'%hooks[2]]
    layer_4 = flex_return_dict['im_feat_dict']['im_feat_%d'%hooks[3]]

    if if_print:
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
            print('--->', layer_4.shape)
        else:
            print('--->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
            pass
        else:
            print(pretrained.act_postprocess1[3:])
            print(pretrained.act_postprocess2[3:])
            print(pretrained.act_postprocess3[3:])
        print(pretrained.act_postprocess4[3:])

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
        layer_1, layer_2, layer_3 = None, None, None
    else:
        layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
        layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
        layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    if if_print:
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
            print('---->', layer_4.shape)
        else:
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

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_perpixel_abs_pos_embed:
        # print(im_feat_init.shape, self.pos_embed_per_pixel.unsqueeze(0).shape)
        im_feat_init += self.pos_embed_per_pixel

    
    batch_size, d = im_feat_init.shape[0], im_feat_init.shape[1]
    spixel_dims = [im_height//self.patch_size[0], im_width//self.patch_size[1]]

    ssn_op = SSNFeatsTransformAdaptive(None, spixel_dims=spixel_dims, if_dense=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_dense)
    # print(spixel_dims)

    mask_resized = input_dict_extra['brdf_loss_mask']

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_from == 'matseg':
        ssn_return_dict = ssn_op(tensor_to_transform=im_feat_init, feats_in=input_dict_extra['return_dict_matseg']['embedding'], mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1, 1./4.)) # Q: [im_height, im_width]
    else:
        ssn_return_dict = ssn_op(tensor_to_transform=im_feat_init, feats_in=pretrained_activations['feat_stage_2'], mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1./2., 1)) # Q: [im_height/4, im_width/4]

    abs_affinity = ssn_return_dict['abs_affinity'] # fixed for now
    # print(abs_affinity.shape, abs_affinity[0].sum(-1).sum(-1), abs_affinity[0].sum(0)) # torch.Size([1, 320, 256, 320]); normalized by **spixel dim (1)**
    dist_matrix = ssn_return_dict['dist_matrix'] # fixed for now
    abs_affinity_normalized_by_pixels = ssn_return_dict['abs_affinity_normalized_by_pixels'] # fixed for now
    
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
    print('====', 'init', torch.mean(im_feat_init), torch.median(im_feat_init), torch.max(im_feat_init), torch.min(im_feat_init))
    proj_coef_dict = {}
    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_reduce_res:
        extra_im_scales = [1., 1., 1., 1.]
    else:
        extra_im_scales = [1., 1./2., 1./2., 1./2.]
    if not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_recompute_C:
        abs_affinity_list = []
        affinity_scales = [1./4., 1./8., 1./16., 1./32.] if not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_reduce_res else [1./4., 1./4., 1./4., 1./4.]
        # affinity_scales = [1./4., 1./4., 1./4., 1./4.]
        for scale in affinity_scales:
            abs_affinity_resized = F.interpolate(abs_affinity, scale_factor=scale, mode='bilinear') / scale / scale
            abs_affinity_resized = abs_affinity_resized / (torch.sum(abs_affinity_resized, 1, keepdims=True)+1e-6)
            abs_affinity_list.append(abs_affinity_resized)
    abs_affinity_normalized_by_pixels_input = F.interpolate(abs_affinity_normalized_by_pixels, scale_factor=1./4., mode='bilinear') * 4. * 4.

    extra_im_scale_accu = 1.
    abs_affinity_idx = 0
    im_feat_idx_recent = im_feat_init

    # [if use im_feat_-1]
    if_use_init_img_feat = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_use_init_img_feat

    for idx, blk in enumerate(self.blocks):
        # print('=asdfasdfsdfasd', x.shape)
        # print(blk[0], blk[0](x).shape)
        x = blk(x) # [-1, 321, 768]
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_slim and idx not in hooks:
            im_feat_dict['im_feat_%d'%idx] = im_feat_idx_recent
            continue

        x_cls_token = x[:, 0:1, :] # [-1, 768, 1]
        x_tokens = x[:, 1:, :].transpose(1, 2) # [-1, 768, 320]

        if idx in hooks:
            idx_within_hooks = hooks.index(idx)
            extra_im_scale = extra_im_scales[idx_within_hooks]
            if not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_reduce_res:
                extra_im_scale_accu *= extra_im_scale
            if idx != 0:
                abs_affinity_idx += 1
        else:
            extra_im_scale = 1.
            # [if use im_feat_-1]
            if if_use_init_img_feat:
                continue

    
        # [if use im_feat_-1]
        # proj_coef_in = abs_affinity_list[abs_affinity_idx]
        # print(spixel_pixel_mul.shape)

        batch_size, sspixels = dist_matrix.shape[:2]
        proj_coef_in = F.interpolate(-dist_matrix.reshape(batch_size, sspixels, abs_affinity.shape[2], abs_affinity.shape[3]), scale_factor=1./4., mode='bilinear')

        # batch_size, sspixels = spixel_pixel_mul.shape[:2]
        # proj_coef_in = F.interpolate(spixel_pixel_mul.reshape(batch_size, sspixels, abs_affinity.shape[2], abs_affinity.shape[3]), scale_factor=1./4., mode='bilinear')

        if if_use_init_img_feat:
            im_feat_idx, proj_coef_idx = ca_modules['layer_%d_ca'%idx](im_feat_dict['im_feat_-1'], x_tokens, im_feat_scale_factor=extra_im_scale_accu, proj_coef_in=proj_coef_in) # torch.Size([1, 768, 320])
        else:
            im_feat_idx, proj_coef_idx = ca_modules['layer_%d_ca'%idx](im_feat_dict['im_feat_%d'%(idx-1)], x_tokens, im_feat_scale_factor=extra_im_scale, proj_coef_in=proj_coef_in) # torch.Size([1, 768, 320])
        # print(idx, extra_im_scale_accu, im_feat_dict['im_feat_%d'%(idx-1)].shape, extra_im_scale, im_feat_idx.shape)
        im_feat_dict['im_feat_%d'%idx] = im_feat_idx
        # if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used and idx!=hooks[-1]:
        #     im_feat_idx = im_feat_idx.detach
        proj_coef_dict['proj_coef_%d'%idx] = proj_coef_idx
        im_feat_idx_recent = im_feat_idx

        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_recompute_C:
            pass
        else:
            assert opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_from == 'matseg', 'only supporting this now'
            assert opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_reduce_res
            # ssn_return_dict_idx = ssn_op(tensor_to_transform=im_feat_idx, affinity_in=abs_affinity_list[abs_affinity_idx], mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1, 1.), if_assert_no_scale=True) # Q: [im_height, im_width]
            # print(abs_affinity_normalized_by_pixels_input.shape, torch.max(abs_affinity_normalized_by_pixels_input), torch.min(abs_affinity_normalized_by_pixels_input), abs_affinity_normalized_by_pixels_input.sum(-1).sum(-1))
            # print(abs_affinity_normalized_by_pixels.shape, torch.max(abs_affinity_normalized_by_pixels), torch.min(abs_affinity_normalized_by_pixels), abs_affinity_normalized_by_pixels.sum(-1).sum(-1))
            ssn_return_dict_idx = ssn_op(tensor_to_transform=im_feat_idx, affinity_in=abs_affinity_normalized_by_pixels_input, mask=mask_resized, if_return_codebook_only=True, \
                scale_down_gamma_tensor=(1, 1.), if_assert_no_scale=True, if_affinity_normalized_by_pixels=True) # Q: [im_height, im_width]

            # print(abs_affinity_list[abs_affinity_idx].shape, proj_coef_idx.shape) # torch.Size([1, 320, 64, 80]) torch.Size([1, 2, 5120, 320])
            # a = abs_affinity_list[abs_affinity_idx]
            # print(a.shape, a.sum(-1).sum(-1)) # torch.Size([1, 320, 64, 80])
            # [if use im_feat_-1]
            # ssn_return_dict_idx = ssn_op(tensor_to_transform=im_feat_idx, affinity_in=abs_affinity_list[abs_affinity_idx], mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1, 1.), if_assert_no_scale=True) # Q: [im_height, im_width]
            c_idx = ssn_return_dict_idx['C']
            x_prime = c_idx.flatten(2).transpose(1, 2) # torch.Size([8, 320, 768]); will be Identity op in qkv recon
            x_prime = torch.cat((x_cls_token, x_prime), dim=1)
            # print('---',torch.mean(x), torch.median(x), torch.max(x), torch.min(x))
            x = x_prime
            if idx == 0 or idx == len(self.blocks)-1:
                print(idx)
                print('====', idx, torch.mean(im_feat_idx), torch.median(im_feat_idx), torch.max(im_feat_idx), torch.min(im_feat_idx), torch.var(im_feat_idx, unbiased=False))
                print('--->',torch.mean(x), torch.median(x), torch.max(x), torch.min(x))
            # print('--->>>',torch.mean(self.norm(x)), torch.median(self.norm(x)), torch.max(self.norm(x)), torch.min(self.norm(x)))
            # x = self.norm(x)
            

        # print(c_idx.shape, x_prime.shape) # torch.Size([1, 768, 320]) torch.Size([1, 320, 768])

    # print('++++', self.norm, self.blocks)
    '''
    # LayerNorm((768,), eps=1e-06, elementwise_affine=True) 
    # Sequential(
      (0): Block(
        (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
    )
    '''
    x = self.norm(x) # LayerNorm

    extra_return_dict = {'Q': ssn_return_dict['Q'], 'matseg_affinity': ssn_return_dict['Q_2D'], 'im_feat': im_feat_init, 'im_feat_dict': im_feat_dict, 'hooks': hooks, 'proj_coef_dict': proj_coef_dict}

    return x, extra_return_dict