import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F
import time
import models_def.models_brdf as models_brdf # basic model

from .vit_SSN_unet_qkv import (
    forward_flex_SSN_unet_qkv_yogo
)

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


def forward_vit_SSN(opt, pretrained, x, input_dict_extra={}):
    b, c, h, w = x.shape

    glob, ssn_return_dict = pretrained.model.forward_flex_SSN(opt, x, pretrained.activations, input_dict_extra=input_dict_extra)

    if (opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_feat_in_transformer) and not opt.cfg.MODEL_BRDF.DPT_baseline.use_vit_only:
        layer_1 = ssn_return_dict['unet_output_dict']['dx4']
        layer_2 = ssn_return_dict['unet_output_dict']['dx3']
    else:
        layer_1 = pretrained.activations["1"]
        layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    # backbone_feat_proj = pretrained.activations['feat_proj']

    if_print = False

    # print(pretrained.activations["stem"].shape) # torch.Size([8, 64, 64, 80])

    # [layer_3 and layer_4 are from transformer layers]
    # print(x.shape)
    if if_print:
        print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # hybrid-SSN: torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768])
    # hybrid-SSN-qkv-unet: torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 321, 768]) torch.Size([1, 321, 768])

    # print(pretrained.act_postprocess1[0:2])
    
    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    if if_print:
        print('->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # hybrid-SSN: -> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 320]) torch.Size([2, 768, 320])
    # hybrid-SSN-qkv: -> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 320]) torch.Size([1, 768, 320])
    # hybrid-SSN-qkv-unet: -> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 320]) torch.Size([1, 768, 320])

    assert pretrained.model.patch_size[0]==pretrained.model.patch_size[1]

    ssn_from = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_from
    recon_method = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_recon_method
    
    if recon_method == 'qtc':
        gamma = ssn_return_dict['Q'] # [b, J, N] for dpt_hybrid.ssn_from = 'matseg'; # [b, J, N/4/4] for dpt_hybrid.ssn_from = 'backbone'
        if layer_1.ndim == 3:
            # layer_1 = unflatten(layer_1)
            if ssn_from == 'matseg':
                layer_1 = QtC(layer_1, gamma, h, w, Q_downsample_rate=4) # reassemble to [b, D, spixel_h, spixel_w]
                # layer_1 = QtC(layer_1, gamma, h, w, Q_downsample_rate=16) # reassemble to [b, D, spixel_h, spixel_w]
            elif ssn_from == 'backbone':
                # layer_1 = QtC(layer_1, gamma, h//4, w//4, Q_downsample_rate=1) # reassemble to [b, D, spixel_h, spixel_w]
                layer_1 = QtC(layer_1, gamma, h//4, w//4, Q_downsample_rate=4) # reassemble to [b, D, spixel_h, spixel_w]
            else:
                assert False, 'invalid ssn_from!'
        if layer_2.ndim == 3:
            # layer_2 = unflatten(layer_2)
            if ssn_from == 'matseg':
                layer_2 = QtC(layer_2, gamma, h, w, Q_downsample_rate=8) # reassemble to [b, D, spixel_h, spixel_w]
                # layer_2 = QtC(layer_2, gamma, h, w, Q_downsample_rate=16) # reassemble to [b, D, spixel_h, spixel_w]
            elif ssn_from == 'backbone':
                # layer_2 = QtC(layer_2, gamma, h//4, w//4, Q_downsample_rate=2) # reassemble to [b, D, spixel_h, spixel_w]
                layer_2 = QtC(layer_2, gamma, h//4, w//4, Q_downsample_rate=4) # reassemble to [b, D, spixel_h, spixel_w]
            else:
                assert False, 'invalid ssn_from!'
        if layer_3.ndim == 3:
            # layer_3 = unflatten(layer_3)
            if ssn_from == 'matseg':
                layer_3 = QtC(layer_3, gamma, h, w, Q_downsample_rate=16) # reassemble to [b, D, spixel_h, spixel_w]
            elif ssn_from == 'backbone':
                layer_3 = QtC(layer_3, gamma, h//4, w//4, Q_downsample_rate=4) # reassemble to [b, D, spixel_h, spixel_w]
            else:
                assert False, 'invalid ssn_from!'
        if layer_4.ndim == 3:
            # layer_4 = unflatten(layer_4)
            if ssn_from == 'matseg':
                layer_4 = QtC(layer_4, gamma, h, w, Q_downsample_rate=32) # reassemble to [b, D, spixel_h, spixel_w]
                # layer_4 = QtC(layer_4, gamma, h, w, Q_downsample_rate=16) # reassemble to [b, D, spixel_h, spixel_w]
            elif ssn_from == 'backbone':
                layer_4 = QtC(layer_4, gamma, h//4, w//4, Q_downsample_rate=4) # reassemble to [b, D, spixel_h, spixel_w]
            else:
                assert False, 'invalid ssn_from!'
    elif recon_method in ['qkv']:
        ca_modules = input_dict_extra['ca_modules']
        extra_im_scales = 1.
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone:
            extra_im_scales = 0.25
        # print(backbone_feat_proj.shape, ssn_return_dict['im_feat'].shape, '++++++')
        if layer_1.ndim == 3:
            # print(layer_1.shape, ssn_return_dict['im_feat'].shape) # torch.Size([1, 768, 320]) torch.Size([1, 1344, 64, 80])
            layer_1 = ca_modules['layer_1_ca'](ssn_return_dict['im_feat'], layer_1, im_feat_scale_factor=extra_im_scales * 1.) # torch.Size([1, 768, 320])
        if layer_2.ndim == 3:
            layer_2 = ca_modules['layer_2_ca'](ssn_return_dict['im_feat'], layer_2, im_feat_scale_factor=extra_im_scales * 1./2.) # torch.Size([1, 768, 320])
        if layer_3.ndim == 3:
            # print(layer_3.shape, ssn_return_dict['im_feat'].shape) # torch.Size([1, 768, 320]) torch.Size([1, 1344, 64, 80])
            layer_3 = ca_modules['layer_3_ca'](ssn_return_dict['im_feat'], layer_3, im_feat_scale_factor=extra_im_scales * 1./4.) # torch.Size([1, 768, 320])
        if layer_4.ndim == 3:
            layer_4 = ca_modules['layer_4_ca'](ssn_return_dict['im_feat'], layer_4, im_feat_scale_factor=extra_im_scales * 1./8.) # torch.Size([1, 768, 320])
    else:
        assert False

    if if_print:
        print('-->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # hybrid-SSN: --> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 16, 20]) torch.Size([2, 768, 16, 20])
    # hybrid-SSN-qkv: --> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 16, 20]) torch.Size([1, 768, 8, 10])
    # hybrid-SSN-qkv-unet: --> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 16, 20]) torch.Size([1, 768, 8, 10])

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    if if_print:
        print('---->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # hybrid-SSN: ----> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 16, 20]) torch.Size([2, 768, 8, 10])
    # hybrid-SSN-qkv-unet: ----> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 768, 16, 20]) torch.Size([1, 768, 8, 10])

    return layer_1, layer_2, layer_3, layer_4, ssn_return_dict


# def _resize_pos_embed_SSN_unet(self, posemb, gs_h, gs_w): # original at https://github.com/rwightman/pytorch-image-models/blob/72b227dcf57c0c62291673b96bdc06576bb90457/timm/models/vision_transformer.py#L481
#     posemb_tok, posemb_grid = (
#         posemb[:, : self.start_index],
#         posemb[0, self.start_index :],
#     )

#     gs_old = int(math.sqrt(len(posemb_grid)))

#     posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
#     posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
#     posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

#     posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

#     return posemb


def forward_flex_SSN_unet(self, opt, x, pretrained_activations=[], input_dict_extra={}):
    b, c, im_height, im_width = x.shape # image pixel space

    # pos_embed = self._resize_pos_embed_SSN_unet(
    #     self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    # )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        # print(self.patch_embed.backbone.forward_features(x).shape, '====')

        output_resnet = self.patch_embed.backbone(x) # [patch_embed] https://github.com/rwightman/pytorch-image-models/blob/72b227dcf57c0c62291673b96bdc06576bb90457/timm/models/layers/patch_embed.py#L15
        # print(_.shape) # hybrid: torch.Size([-1, 1024, 16, 20])
        # if isinstance(x, (list, tuple)):
        #     x = x[-1]  # last feature if backbone outputs list/tuple of features

    # for key in ["feat_stem", "feat_stage_0", "feat_stage_1", "feat_stage_2"]:
    #     print(key, pretrained_activations[key].shape, input_dict_extra.keys())
        # feat_stem torch.Size([8, 64, 64, 80])
        # stage_0 torch.Size([8, 256, 64, 80])
        # stage_1 torch.Size([8, 512, 32, 40])
        # stage_2 torch.Size([8, 512, 32, 40])
    
    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone:
        x_renormalzied = x

        # mean_1 = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
        # std_1 = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
        # mean_2 = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
        # std_2 = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).reshape(1, 3, 1, 1).cuda()

        # x_renormalzied = ((x * std_1 + mean_1) - mean_2) / std_2
        x1, x2, x3, x4, x5, x6, _ = self.patch_embed.unet_backbone.encoder(x_renormalzied, input_dict_extra={})
        unet_output_dict = self.patch_embed.unet_backbone.albedoDecoder(x_renormalzied, x1, x2, x3, x4, x5, x6, input_dict_extra={})
        albedo_dx6 = unet_output_dict['dx6']
        '''
        dx1 torch.Size([1, 512, 8, 10])
        dx2 torch.Size([1, 256, 16, 20])
        dx3 torch.Size([1, 256, 32, 40])
        dx4 torch.Size([1, 128, 64, 80])
        dx5 torch.Size([1, 64, 128, 160])
        dx6 torch.Size([1, 64, 256, 320])
        '''
        # for key in ['dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6']:
        #     print(key, unet_output_dict[key].shape)

        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_debug_unet:
            albedo_pred_unet = 0.5 * (unet_output_dict['x_out'] + 1)
            albedo_pred_unet = input_dict_extra['input_dict']['segBRDFBatch'] * albedo_pred_unet
            # albedo_pred_unet = models_brdf.LSregress(albedo_pred_unet * input_dict_extra['input_dict']['segBRDFBatch'].expand_as(albedo_pred_unet),
            #         input_dict_extra['input_dict']['albedoBatch'] * input_dict_extra['input_dict']['segBRDFBatch'].expand_as(input_dict_extra['input_dict']['albedoBatch']), albedo_pred_unet)
            albedo_pred_unet = torch.clamp(albedo_pred_unet ** (1.0/2.2), 0, 1)

            print(albedo_pred_unet.shape, torch.max(albedo_pred_unet),torch.min(albedo_pred_unet), torch.median(albedo_pred_unet))

    # print(h, w, self.patch_size)
    assert all([x!=1 for x in self.patch_size])
    assert im_height%self.patch_size[0]==0 and im_width%self.patch_size[1]==0

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone:
        im_feat = albedo_dx6
    elif opt.cfg.MODEL_BRDF.DPT_baseline.model=='dpt_large_SSN':
        im_feat = pretrained_activations['feat_stage_0']
    else:
        im_feat = torch.cat(
            [
                F.interpolate(pretrained_activations['feat_stem'], scale_factor=1, mode='bilinear'), # torch.Size([4, 64, 64, 80])
                F.interpolate(pretrained_activations['feat_stage_0'], scale_factor=1, mode='bilinear'), # torch.Size([4, 256, 64, 80])
                F.interpolate(pretrained_activations['feat_stage_1'], scale_factor=2, mode='bilinear'), # torch.Size([4, 512, 32, 40])
                F.interpolate(pretrained_activations['feat_stage_2'], scale_factor=2, mode='bilinear'), # torch.Size([4, 512, 32, 40])
                F.interpolate(output_resnet, scale_factor=4, mode='bilinear'), # torch.Size([4, 1024, 16, 20])
            ], dim=1)
        im_feat = self.patch_embed.proj_extra(im_feat)
        # print(im_feat.shape, '+++++++++++++') # torch.Size([1, 768, 64, 80])
    # print(im_feat.shape, input_dict_extra['return_dict_matseg']['embedding'].shape) #  # torch.Size([4, D, 64, 80])
    
    # if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_from == 'matseg':
    #     assert im_feat.shape[-2:] == input_dict_extra['return_dict_matseg']['embedding'].shape[-2:]
    batch_size, d = im_feat.shape[0], im_feat.shape[1]
    spixel_dims = [im_height//self.patch_size[0], im_width//self.patch_size[1]]

    ssn_op = SSNFeatsTransformAdaptive(None, spixel_dims=spixel_dims, if_dense=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_dense)

    # print(input_dict_extra['brdf_loss_mask'].shape, im_feat.shape, input_dict_extra['return_dict_matseg']['embedding'].shape)
    # mask_resized = F.interpolate(input_dict_extra['brdf_loss_mask'].unsqueeze(1), scale_factor=1./4., mode='nearest').squeeze(1) # torch.Size([1, 256, 320]) torch.Size([1, 1344, 64, 80]) torch.Size([1, 4, 256, 320])
    mask_resized = input_dict_extra['brdf_loss_mask']
    # print(input_dict_extra['brdf_loss_mask'].unsqueeze(1).shape, mask_resized.shape)

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone:
        ssn_return_dict = ssn_op(tensor_to_transform=im_feat, feats_in=input_dict_extra['return_dict_matseg']['embedding'], mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1, 1.)) # Q: [im_height, im_width]
    else:
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_from == 'matseg':
            ssn_return_dict = ssn_op(tensor_to_transform=im_feat, feats_in=input_dict_extra['return_dict_matseg']['embedding'], mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1, 1./4.)) # Q: [im_height, im_width]
        else:
            ssn_return_dict = ssn_op(tensor_to_transform=im_feat, feats_in=pretrained_activations['feat_stage_2'].detach(), mask=mask_resized, if_return_codebook_only=True, scale_down_gamma_tensor=(1./2., 1)) # Q: [im_height/4, im_width/4]
    
    c = ssn_return_dict['C'] # codebook
    c = c.view([batch_size, d, spixel_dims[0], spixel_dims[1]])

    # print(ssn_return_dict['C'].shape, ssn_return_dict['Q_2D'].shape) # torch.Size([2, 1344, 320]) torch.Size([2, 320, 256, 320])
    x = self.patch_embed.proj(c).flatten(2).transpose(1, 2) # torch.Size([8, 320, 768]); will be Identity op in qkv recon

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
    # x = self.pos_drop(x) # simply Dropout layer
    for idx, blk in enumerate(self.blocks):
        x = blk(x) # always [8, 321, 768]

    x = self.norm(x)

    extra_return_dict = {'Q': ssn_return_dict['Q'], 'matseg_affinity': ssn_return_dict['Q_2D'], 'im_feat': im_feat}

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone:
        extra_return_dict.update({'unet_output_dict': unet_output_dict})
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_debug_unet:
            extra_return_dict.update({'albedo_pred_unet': albedo_pred_unet})

    return x, extra_return_dict

def _make_vit_b_rn50_backbone_SSN_unet(
    opt, 
    model,
    features=[256, 512, 768, 768],
    size=[384, 384],
    hooks=[0, 1, 8, 11],
    vit_features=768,
    use_vit_only=False,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False):
    pretrained = nn.Module()

    pretrained.model = model

    if not(opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_feat_in_transformer):
        # in this case, no need for resnet backbones
        pretrained.model.patch_embed.backbone.stem.register_forward_hook(
            get_activation("feat_stem")
        )
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
            get_activation("feat_stage_0")
        )
        if not opt.cfg.MODEL_BRDF.DPT_baseline.model=='dpt_large_SSN':
            pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
                get_activation("feat_stage_1")
            )
            pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
                get_activation("feat_stage_2")
            )
        
    pretrained.model.patch_embed.proj.register_forward_hook(
        get_activation("feat_proj")
    )

    if use_vit_only == True:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    else:
        if not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_feat_in_transformer:
            pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
                get_activation("1")
            )
            pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
                get_activation("2")
            )
            # pass

    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    if enable_attention_hooks:
        pretrained.model.blocks[2].attn.register_forward_hook(get_attention("attn_1"))
        pretrained.model.blocks[5].attn.register_forward_hook(get_attention("attn_2"))
        pretrained.model.blocks[8].attn.register_forward_hook(get_attention("attn_3"))
        pretrained.model.blocks[11].attn.register_forward_hook(get_attention("attn_4"))
        pretrained.attention = attention

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    recon_method = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_recon_method
        
    if use_vit_only == True:
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
            )
        ]
        if recon_method in ['qkv', 'qtc']:
            act_postprocess1_list.pop()
            # if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_reduce_res:
                # act_postprocess1_list.pop() # not sizing up
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
            # if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_reduce_res:
                # act_postprocess2_list.pop() # not sizing up

        pretrained.act_postprocess2 = nn.Sequential(*act_postprocess2_list)
    else:
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
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_reduce_res:
            act_postprocess4_list.pop() # not sizing up

    pretrained.act_postprocess4 = nn.Sequential(*act_postprocess4_list)

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.patch_size]*2

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv and (recon_method == 'qkv'):
        pretrained.model.forward_flex_SSN = types.MethodType(forward_flex_SSN_unet_qkv_yogo, pretrained.model)
    else:
        pretrained.model.forward_flex_SSN = types.MethodType(forward_flex_SSN_unet, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    # pretrained.model._resize_pos_embed_SSN = types.MethodType(
    #     _resize_pos_embed_SSN, pretrained.model
    # )

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
        pretrained.act_postprocess1  = nn.Identity()
        pretrained.act_postprocess2  = nn.Identity()
        pretrained.act_postprocess3  = nn.Identity()

    return pretrained

def _make_vit_b16_backbone_SSN_unet(
    opt, 
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,):

    assert False

    pretrained = nn.Module()

    pretrained.model = model
    

    pretrained.model.patch_embed.backbone.stem.register_forward_hook(
        get_activation("feat_stem")
    )
    pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
        get_activation("feat_stage_0")
    )
    pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
        get_activation("feat_stage_1")
    )
    pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
        get_activation("feat_stage_2")
    )


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

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
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
    )

    pretrained.act_postprocess2 = nn.Sequential(
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

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.patch_size]*2

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex_SSN = types.MethodType(forward_flex_SSN_unet, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    # pretrained.model._resize_pos_embed_SSN = types.MethodType(
    #     _resize_pos_embed_SSN, pretrained.model
    # )

    return pretrained

def _make_pretrained_vitb_unet_384_SSN(
    opt, 
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
):
    # [DPT-SSN model] (v1) https://i.imgur.com/iSmi5wt.png
    print('========= [hybrid-SSN] [_make_pretrained_vitb_unet_384_SSN] pretrained', pretrained)

    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)
    # model = create_model_patch_embed_unet(opt)
    # model.patch_embed = create_model_patch_embed_unet(opt).patch_embed
    # model.patch_embed = HybridEmbed(
        # img_size=(opt.cfg.DATA.im_height, opt.cfg.DATA.im_width), patch_size=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.patch_size, in_chans=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.backbone_dims, embed_dim=768)
    # patch_size = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.patch_size


    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone:
        import models_def.models_brdf as models_brdf # basic model
        encoder_to_use = models_brdf.encoder0
        decoder_to_use = models_brdf.decoder0
        unet_backbone = nn.ModuleDict(
            {
                'encoder': encoder_to_use(opt, cascadeLevel = 0, in_channels = 3), 
                'albedoDecoder': decoder_to_use(opt, mode=0, modality='al', if_not_final_fc=not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_debug_unet)
            }
        )
        model.patch_embed.unet_backbone = unet_backbone

        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_feat_in_transformer:
            # no need to keep the resnet backbone then
            model.patch_embed.backbone = nn.Identity()


    # print(model)

    # model.patch_embed.proj = nn.Conv2d(opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.backbone_dims, opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.feat_proj_channels, kernel_size=1, stride=1)
    model.patch_embed.proj = nn.Identity()

    backbone_dims = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.backbone_dims
    feat_proj_channels = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.feat_proj_channels

    if_batch_norm_in_proj_extra = opt.cfg.MODEL_BRDF.DPT_baseline.if_batch_norm_in_proj_extra_in_proj_extra

    if opt.cfg.MODEL_BRDF.DPT_baseline.if_simple_proj_extra:
        model.patch_embed.proj_extra = nn.Conv2d(backbone_dims, feat_proj_channels, kernel_size=1, stride=1)
    else:
        model.patch_embed.proj_extra = nn.Sequential(
            nn.Conv2d(backbone_dims, backbone_dims//2, kernel_size=1, stride=1), 
            nn.BatchNorm2d(backbone_dims//2) if if_batch_norm_in_proj_extra else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(backbone_dims//2, feat_proj_channels, kernel_size=1, stride=1), 
            nn.BatchNorm2d(feat_proj_channels) if if_batch_norm_in_proj_extra else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(feat_proj_channels, feat_proj_channels, kernel_size=1, stride=1)
        )

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_perpixel_abs_pos_embed:
        # similar to https://github.com/rwightman/pytorch-image-models/blob/72b227dcf57c0c62291673b96bdc06576bb90457/timm/models/vision_transformer.py#L271
        model.pos_embed_per_pixel = nn.Parameter(torch.zeros(1, feat_proj_channels, 256//4, 320//4))
        from timm.models.layers import trunc_normal_
        trunc_normal_(model.pos_embed_per_pixel, std=.02)

    # [def vit_base_r50_s16_384()] https://github.com/rwightman/pytorch-image-models/blob/79927baaecb6cdd1a25eed7f0f8c122b99712c72/timm/models/vision_transformer_hybrid.py#L232
    # [def _create_vision_transformer()] https://github.com/rwightman/pytorch-image-models/blob/79927baaecb6cdd1a25eed7f0f8c122b99712c72/timm/models/vision_transformer.py#L513
    # [class HybridEmbed(nn.Module)] (Extract feature map from CNN, flatten, project to embedding dim.）https://github.com/rwightman/pytorch-image-models/blob/79927baaecb6cdd1a25eed7f0f8c122b99712c72/timm/models/vision_transformer_hybrid.py#L100
    # [resnetv2'] https://github.com/rwightman/pytorch-image-models/blob/766b4d32627fc4d1d9d188de81736504215127a0/timm/models/resnetv2.py#L338

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    if_unet_feat_in_transformer = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_feat_in_transformer

    return _make_vit_b_rn50_backbone_SSN_unet(
        opt, 
        model,
        features=[256, 512, 768, 768] if not if_unet_feat_in_transformer else [128, 256, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_vitl_unet_384_SSN(
    opt, 
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=True,
    enable_attention_hooks=False):

    assert use_vit_only
    # assert False, 'for DPT-large but not ready'
    # [DPT-SSN model] (v1) https://i.imgur.com/iSmi5wt.png
    print('========= [large-SSN] [_make_pretrained_vitl_unet_384_SSN] pretrained', pretrained)
    model_1 = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)
    resnet_backbone = model_1.patch_embed
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)
    model.patch_embed = resnet_backbone

    # print(model)

    # model = create_model_patch_embed_unet(opt)
    # model.patch_embed = create_model_patch_embed_unet(opt).patch_embed
    # model.patch_embed = HybridEmbed(
        # img_size=(opt.cfg.DATA.im_height, opt.cfg.DATA.im_width), patch_size=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.patch_size, in_chans=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.backbone_dims, embed_dim=768)
    # patch_size = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.patch_size
    model.patch_embed.proj = nn.Conv2d(opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.backbone_dims, opt.cfg.MODEL_BRDF.DPT_baseline.dpt_large.feat_proj_channels, kernel_size=1, stride=1)

    # print(model.patch_embed.backbone.stages)
    model.patch_embed.backbone.stages[1] = nn.Identity()
    model.patch_embed.backbone.stages[2] = nn.Identity()

    # if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used:
    #     model.patch_embed.backbone.stages[2] = nn.Identity()


    # [def vit_base_r50_s16_384()] https://github.com/rwightman/pytorch-image-models/blob/79927baaecb6cdd1a25eed7f0f8c122b99712c72/timm/models/vision_transformer_hybrid.py#L232
    # [def _create_vision_transformer()] https://github.com/rwightman/pytorch-image-models/blob/79927baaecb6cdd1a25eed7f0f8c122b99712c72/timm/models/vision_transformer.py#L513
    # [class HybridEmbed(nn.Module)] (Extract feature map from CNN, flatten, project to embedding dim.）https://github.com/rwightman/pytorch-image-models/blob/79927baaecb6cdd1a25eed7f0f8c122b99712c72/timm/models/vision_transformer_hybrid.py#L100
    # [resnetv2'] https://github.com/rwightman/pytorch-image-models/blob/766b4d32627fc4d1d9d188de81736504215127a0/timm/models/resnetv2.py#L338

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b_rn50_backbone_SSN_unet(
        opt, 
        model,
        features=[256, 512, 1024, 1024],
        size=[384, 384],
        hooks=hooks,
        vit_features=1024, 
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
