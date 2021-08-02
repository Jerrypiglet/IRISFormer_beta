import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F

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

def QtC(codebook, gamma, im_height, im_width, Q_downsample_rate=1):
    '''
    C: [B, D, J]
    gamma: [B, J, N]
    '''
    assert len(gamma.shape)==3
    batch_size, J, N = gamma.shape
    D = codebook.shape[1]
    assert im_height * im_width == N

    if Q_downsample_rate != 1:
        gamma_resampled = gamma.view(batch_size, J, im_height, im_width)
        gamma_resampled = F.interpolate(gamma_resampled, scale_factor=1./float(Q_downsample_rate))
        gamma_resampled = gamma_resampled / (torch.sum(gamma_resampled, 1, keepdims=True)+1e-6)
        gamma_resampled = gamma_resampled.view(batch_size, J, -1)
    else:
        gamma_resampled = gamma

    # print(Q_downsample_rate, codebook.shape, gamma_resampled.shape)
    im_hat = codebook @ gamma_resampled
    im_hat = im_hat.view(batch_size, D, im_height//Q_downsample_rate, im_width//Q_downsample_rate)
    return im_hat


def forward_vit_SSN(opt, pretrained, x, input_dict_extra={}):
    b, c, h, w = x.shape

    glob, ssn_return_dict = pretrained.model.forward_flex_SSN(x, pretrained.activations, input_dict_extra=input_dict_extra)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    # print(pretrained.activations["stem"].shape) # torch.Size([8, 64, 64, 80])

    # [layer_3 and layer_4 are from transformer layers]
    # print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape) # torch.Size([8, 256, 64, 80]) torch.Size([8, 512, 32, 40]) torch.Size([8, 321, 768]) torch.Size([8, 321, 768])

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    # print('->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape) # -> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 320]) torch.Size([2, 768, 320])

    # unflatten = nn.Sequential( # 'Re-assemble' in DPT paper
    #     nn.Unflatten(
    #         2,
    #         torch.Size(
    #             [
    #                 h // pretrained.model.patch_size[1],
    #                 w // pretrained.model.patch_size[0],
    #             ]
    #         ),
    #     )
    # )

    gamma = ssn_return_dict['Q'] # [b, J, N]
    # print(Q.shape)

    if layer_1.ndim == 3:
        # layer_1 = unflatten(layer_1)
        layer_1 = QtC(layer_1, gamma, h, w, Q_downsample_rate=4)
    if layer_2.ndim == 3:
        # layer_2 = unflatten(layer_2)
        layer_2 = QtC(layer_2, gamma, h, w, Q_downsample_rate=8)
    if layer_3.ndim == 3:
        # layer_3 = unflatten(layer_3)
        layer_3 = QtC(layer_3, gamma, h, w, Q_downsample_rate=16)
    if layer_4.ndim == 3:
        # layer_4 = unflatten(layer_4)
        layer_4 = QtC(layer_4, gamma, h, w, Q_downsample_rate=16)

    # print('-->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape) # --> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 16, 20]) torch.Size([2, 768, 16, 20])

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    # print('---->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape) # --> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 16, 20]) torch.Size([2, 768, 8, 10])


    return layer_1, layer_2, layer_3, layer_4


def _resize_pos_embed_SSN_unet(self, posemb, gs_h, gs_w): # original at https://github.com/rwightman/pytorch-image-models/blob/72b227dcf57c0c62291673b96bdc06576bb90457/timm/models/vision_transformer.py#L481
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


def forward_flex_SSN_unet(self, x, pretrained_activations=[], input_dict_extra={}):
    b, c, h, w = x.shape # image pixel space

    # pos_embed = self._resize_pos_embed_SSN_unet(
    #     self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    # )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        # print(self.patch_embed.backbone.forward_features(x).shape, '====')
        _ = self.patch_embed.backbone(x) # [patch_embed] https://github.com/rwightman/pytorch-image-models/blob/72b227dcf57c0c62291673b96bdc06576bb90457/timm/models/layers/patch_embed.py#L15
        # if isinstance(x, (list, tuple)):
        #     x = x[-1]  # last feature if backbone outputs list/tuple of features

    # for key in ["feat_stem", "feat_stage_0", "feat_stage_1", "feat_stage_2"]:
    #     print(key, pretrained_activations[key].shape, input_dict_extra.keys())
        # feat_stem torch.Size([8, 64, 64, 80])
        # stage_0 torch.Size([8, 256, 64, 80])
        # stage_1 torch.Size([8, 512, 32, 40])
        # stage_2 torch.Size([8, 512, 32, 40])
    
    # print(h, w, self.patch_size)
    assert all([x!=1 for x in self.patch_size])
    assert h%self.patch_size[0]==0 and w%self.patch_size[1]==0
    im_feat = torch.cat(
        [
            F.interpolate(pretrained_activations['feat_stem'], scale_factor=4), 
            F.interpolate(pretrained_activations['feat_stage_0'], scale_factor=4), 
            F.interpolate(pretrained_activations['feat_stage_1'], scale_factor=8), 
            F.interpolate(pretrained_activations['feat_stage_2'], scale_factor=8), 
        ], dim=1)
    # print(im_feat.shape, input_dict_extra['return_dict_matseg']['embedding'].shape)

    assert im_feat.shape[-2:] == input_dict_extra['return_dict_matseg']['embedding'].shape[-2:]
    batch_size, d = im_feat.shape[0], im_feat.shape[1]
    spixel_dims = [h//self.patch_size[0], w//self.patch_size[1]]

    ssn_op = SSNFeatsTransformAdaptive(None, spixel_dims=spixel_dims)
    ssn_return_dict = ssn_op(tensor_to_transform=im_feat, feats_in=input_dict_extra['return_dict_matseg']['embedding'], if_return_codebook_only=True)
    c = ssn_return_dict['C'] # codebook
    c = c.view([batch_size, d, spixel_dims[0], spixel_dims[1]])

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

    return x, ssn_return_dict

# def forward_flex_SSN_unet_(self, x, pretrained_activations=[]):
#     b, c, h, w = x.shape # image pixel space

#     # pos_embed = self._resize_pos_embed_SSN_unet(
#     #     self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
#     # )

#     B = x.shape[0]

#     for key in ["feat_stem", "stage_0", "stage_1", "stage_2"]:
#         print(key, pretrained_activations[key].shape)

#     if hasattr(self.patch_embed, "backbone"):
#         # print(self.patch_embed.backbone.forward_features(x).shape, '====')
#         x = self.patch_embed.backbone(x) # [patch_embed] https://github.com/rwightman/pytorch-image-models/blob/72b227dcf57c0c62291673b96bdc06576bb90457/timm/models/layers/patch_embed.py#L15
#         if isinstance(x, (list, tuple)):
#             x = x[-1]  # last feature if backbone outputs list/tuple of features
#         # print(x.shape, self.patch_embed.proj(x).shape) # torch.Size([8, 1024, 16, 20]) torch.Size([8, 768, 16, 20])

#     x = self.patch_embed.proj(x).flatten(2).transpose(1, 2) # torch.Size([8, 320, 768])

#     if getattr(self, "dist_token", None) is not None:
#         cls_tokens = self.cls_token.expand(
#             B, -1, -1
#         )  # stole cls_tokens impl from Phil Wang, thanks
#         dist_token = self.dist_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, dist_token, x), dim=1)
#     else:
#         cls_tokens = self.cls_token.expand(
#             B, -1, -1
#         )  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)

#     # x = x + pos_embed
#     # print(x.shape, pos_embed.shape) # torch.Size([8, 321, 768]) torch.Size([1, 321, 768])
#     x = self.pos_drop(x)

#     for idx, blk in enumerate(self.blocks):
#         x = blk(x) # always [8, 321, 768]

#     x = self.norm(x)

#     return x



def _make_vit_b_rn50_backbone_SSN_unet(
    model,
    features=[256, 512, 768, 768],
    size=[384, 384],
    hooks=[0, 1, 8, 11],
    vit_features=768,
    use_vit_only=False,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
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

    if use_vit_only == True:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    else:

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

    if use_vit_only == True:
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
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex_SSN = types.MethodType(forward_flex_SSN_unet, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    # pretrained.model._resize_pos_embed_SSN = types.MethodType(
    #     _resize_pos_embed_SSN, pretrained.model
    # )

    return pretrained

# class BackboneResnetFeat(Baseline):
#     def forward(self, x):
#         # bottom up
#         c1, c2, c3, c4, c5 = self.backbone(x)
#         # print(x.shape, c1.shape, c2.shape, c3.shape, c4.shape, c5.shape) # torch.Size([8, 3, 240, 320]) torch.Size([8, 128, 120, 160]) torch.Size([8, 256, 60, 80]) torch.Size([8, 512, 30, 40]) torch.Size([8, 1024, 15, 20]) torch.Size([8, 2048, 8, 10])

#         # top down
#         p0, p1, p2, p3, p4, p5 = self.top_down((c1, c2, c3, c4, c5)) # [16, 3, 192, 256],  [16, 64, 96, 128],  [16, 128, 48, 64],  [16, 256, 24, 32],  [16, 256, 12, 16],  [16, 512, 6, 8]
#         # feats_matseg_dict = {'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5}
#         # print(p0.shape, p1.shape, p2.shape, p3.shape, p4.shape, p5.shape)

#         # output
#         # logit = self.pred_prob(p0)
#         embedding = self.embedding_conv(p0)
#         # depth = self.pred_depth(p0)
#         # surface_normal = self.pred_surface_normal(p0)
#         # param = self.pred_param(p0)

#         # return_dict = {'logit': logit, 'embedding': embedding, 'feats_matseg_dict': feats_matseg_dict}
#         # return return_dict
#         # return prob, embedding
#         # return prob, embedding, depth, surface_normal, param

#         return embedding

# def create_model_patch_embed_unet(opt):
#     model = nn.Module()
#     model.patch_embed = nn.Module()
#     model.patch_embed.backbone = BackboneResnetFeat(opt.cfg.MODEL_MATSEG, embed_dims=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid_SSN.backbone_dims, input_dim=3)
#     return model

from timm.models.vision_transformer_hybrid import HybridEmbed


def _make_pretrained_vitb_unet_384_SSN(
    opt, 
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
):
    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)
    # model = create_model_patch_embed_unet(opt)
    # model.patch_embed = create_model_patch_embed_unet(opt).patch_embed
    # model.patch_embed = HybridEmbed(
        # img_size=(opt.cfg.DATA.im_height, opt.cfg.DATA.im_width), patch_size=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid_SSN.patch_size, in_chans=opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid_SSN.backbone_dims, embed_dim=768)
    # patch_size = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid_SSN.patch_size
    model.patch_embed.proj = nn.Conv2d(opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid_SSN.backbone_dims, 768, kernel_size=1, stride=1)

    # [def vit_base_r50_s16_384()] https://github.com/rwightman/pytorch-image-models/blob/79927baaecb6cdd1a25eed7f0f8c122b99712c72/timm/models/vision_transformer_hybrid.py#L232
    # [def _create_vision_transformer()] https://github.com/rwightman/pytorch-image-models/blob/79927baaecb6cdd1a25eed7f0f8c122b99712c72/timm/models/vision_transformer.py#L513
    # [class HybridEmbed(nn.Module)] (Extract feature map from CNN, flatten, project to embedding dim.）https://github.com/rwightman/pytorch-image-models/blob/79927baaecb6cdd1a25eed7f0f8c122b99712c72/timm/models/vision_transformer_hybrid.py#L100
    # [resnetv2'] https://github.com/rwightman/pytorch-image-models/blob/766b4d32627fc4d1d9d188de81736504215127a0/timm/models/resnetv2.py#L338

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_vit_b_rn50_backbone_SSN_unet(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
