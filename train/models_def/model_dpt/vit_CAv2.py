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


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1).contiguous()
        return x


def forward_vit_CAv2(opt, pretrained, x, extra_input_dict={}):
    b, c, h, w = x.shape

    glob, extra_output_dict = pretrained.model.forward_flex(opt, x, pretrained.activations, extra_input_dict=extra_input_dict)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]
    
    if_print = False

    if if_print:
        print(layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # DPT-hybrid: torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 321, 768]) torch.Size([2, 321, 768])
    # DPT-large: torch.Size([1, 321, 1024]) torch.Size([1, 321, 1024]) torch.Size([1, 321, 1024]) torch.Size([1, 321, 1024])

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    if if_print:
        print('->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # DPT-hybrid: -> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 320]) torch.Size([2, 768, 320])
    # DPT-large: -> torch.Size([1, 1024, 320]) torch.Size([1, 1024, 320]) torch.Size([1, 1024, 320]) torch.Size([1, 1024, 320])
    
    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
        im_feat_dict, output_hooks = extra_output_dict['im_feat_dict'], extra_input_dict['output_hooks']
        if if_print:
            print(im_feat_dict.keys(), output_hooks)

    if layer_1.ndim == 3:
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
            layer_1 = im_feat_dict['im_feat_%d'%output_hooks[0]]
            # print('+++')
        else:
            layer_1 = unflatten(layer_1)

    if layer_2.ndim == 3:
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
            layer_2 = im_feat_dict['im_feat_%d'%output_hooks[1]]
            # print('+++')
        else:
            layer_2 = unflatten(layer_2)

    if layer_3.ndim == 3:
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
            layer_3 = im_feat_dict['im_feat_%d'%output_hooks[2]]
            # print('+++')
        else:
            layer_3 = unflatten(layer_3)

    if layer_4.ndim == 3:
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
            layer_4 = im_feat_dict['im_feat_%d'%output_hooks[3]]
            # print('+++')
        else:
            layer_4 = unflatten(layer_4)

    if if_print:
        print('-->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # DPT-hybrid: --> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 16, 20]) torch.Size([2, 768, 16, 20])
    # DPT-large: --> torch.Size([1, 1024, 16, 20]) torch.Size([1, 1024, 16, 20]) torch.Size([1, 1024, 16, 20]) torch.Size([1, 1024, 16, 20])



    # layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    # layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    # layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    # layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    # if if_print:
        # print('---->', layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
    # DPT-hybrid: ----> torch.Size([2, 256, 64, 80]) torch.Size([2, 512, 32, 40]) torch.Size([2, 768, 16, 20]) torch.Size([2, 768, 8, 10])
    # DPT-large: ----> torch.Size([1, 256, 64, 80]) torch.Size([1, 512, 32, 40]) torch.Size([1, 1024, 16, 20]) torch.Size([1, 1024, 8, 10])


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


def forward_flex_CAv2(self, opt, x_input, pretrained_activations=[], extra_input_dict={}):
    b, c, h, w = x_input.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x_input.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x_input)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features; 1/16
    else:
        assert False, 'DPT-hybrid variants have to have a backbone!'

    # print(x.shape) # torch.Size([1, 1024, 16, 20])
    im_feat_resnet = self.patch_embed.proj(x)
    x = im_feat_resnet.flatten(2).transpose(1, 2)


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

    # if opt.cfg.MODEL_BRDF.DPT_baseline.if_pos_embed:
    #     x = x + pos_embed
    # x = self.pos_drop(x)


    # stem for I_feat_init
    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_type in ['single', 'double']:
        I_feat_init = self.stem(x_input) # c 64 or 256, 1/4
    else:
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_shared_stem:
            pass # already forwarded
        else:
            _ = self.stem(x_input)
        I_feat_init = torch.cat(
            [
                F.interpolate(pretrained_activations['stem_feat_stem'], scale_factor=1, mode='bilinear'), # torch.Size([4, 64, 64, 80])
                F.interpolate(pretrained_activations['stem_feat_stage_0'], scale_factor=1, mode='bilinear'), # torch.Size([4, 256, 64, 80])
                F.interpolate(pretrained_activations['stem_feat_stage_1'], scale_factor=2, mode='bilinear'), # torch.Size([4, 512, 32, 40])
                F.interpolate(pretrained_activations['stem_feat_stage_2'], scale_factor=4, mode='bilinear'), # torch.Size([4, 1024, 16, 20])
            ], dim=1)
        I_feat_init = self.stem.proj_extra(I_feat_init) # torch.Size([4, 768, 64, 80])
        # print('----', I_feat_init.shape)

    assert opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA
    im_feat_dict = {}
    im_feat_dict['im_feat_-1'] = I_feat_init
    assert I_feat_init.shape[1] == opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.im_feat_init_c # stem feat dims
    ca_modules = extra_input_dict['ca_modules']
    output_hooks = extra_input_dict['output_hooks']

    im_feat_scale_factor_accu = 1.

    if_print = False

    for idx, blk in enumerate(self.blocks):
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.keep_N_layers!=-1 and idx >= opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.keep_N_layers:
            continue

        x = blk(x)
        if if_print:
            print('=========', idx)
        
        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
            assert opt.cfg.MODEL_BRDF.DPT_baseline.readout == 'ignore'
            # print(x.shape, im_feat_init.shape) # torch.Size([1, 321, 768]) torch.Size([1, 768, 16, 20])
            x_cls_token = x[:, 0:1, :] # [-1, 1, 768]
            x_tokens = x[:, 1:, :].transpose(1, 2) # [-1, 768, 320]

            assert opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_proj_method == 'full'
            # print(idx, ca_modules['layer_%d_ca'%idx])
            
            if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_transform_feat_in_qkv_if_use_init_img_feat:
                if idx in output_hooks[1:]:
                    im_feat_scale_factor_accu *= 1./2.
                im_feat_scale_factor = im_feat_scale_factor_accu
                im_feat_in = im_feat_dict['im_feat_-1']
            else:
                im_feat_in = im_feat_dict['im_feat_%d'%(idx-1)]
                if idx in output_hooks[1:]:
                    im_feat_scale_factor = 1./2.
                else:
                    im_feat_scale_factor = 1.

            if if_print:
                print(im_feat_in.shape, ca_modules['layer_%d_ca'%idx])

            im_feat_out, proj_coef_idx = ca_modules['layer_%d_ca'%idx](im_feat_in, x_tokens, im_feat_scale_factor=im_feat_scale_factor, proj_coef_in=None) # torch.Size([1, 768, 320])

            if if_print:
                print(im_feat_out.shape) # torch.Size([1, 768, 16, 20])

            im_feat_dict['im_feat_%d'%idx] = im_feat_out

            if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA_if_recompute_C:
                assert False, 'not applicable because of different dims of I_feat (256, 512, ...) and tokens (738)'
                x_prime = im_feat_out.flatten(2).transpose(-1, -2)
                x = torch.cat((x_cls_token, x_prime), dim=1)
            
            if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CAc:
                if idx == (len(self.blocks)-1):
                    continue
                if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CAc_if_use_previous_feat:
                    im_feat_in = im_feat_dict['im_feat_%d'%(idx-1)]
                elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CAc_if_use_init_feat:
                    im_feat_in = im_feat_dict['im_feat_-1']
                else:
                    im_feat_in = im_feat_dict['im_feat_%d'%idx]
                im_feat_in = im_feat_in.flatten(2)
                x_tokens_out, _ = ca_modules['layer_%d_cac'%idx](x_tokens, im_feat_in, if_in_feature_flattened=True)
                x = torch.cat((x_cls_token, x_tokens_out.transpose(-1, -2)), dim=1)



        # print('====', idx, torch.mean(x), torch.median(x), torch.max(x), torch.min(x), torch.var(x, unbiased=False))

        # print(x.shape)
        # print(blk)

    x = self.norm(x)

    extra_output_dict = {}
    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA:
        extra_output_dict['im_feat_dict'] = im_feat_dict

    return x, extra_output_dict


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

def _make_vit_b_rn50_backbone_CAv2(
    opt, 
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

    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_type == 'full':
        # in this case, no need for resnet backbones
        if not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_shared_stem:
            pretrained.model.stem.stem.register_forward_hook(
                get_activation("stem_feat_stem")
            )
            pretrained.model.stem.stages[0].register_forward_hook(
                get_activation("stem_feat_stage_0")
            )
            pretrained.model.stem.stages[1].register_forward_hook(
                get_activation("stem_feat_stage_1")
            )
            pretrained.model.stem.stages[2].register_forward_hook(
                get_activation("stem_feat_stage_2")
            )
        else:
            pretrained.model.patch_embed.backbone.stem.register_forward_hook(
                get_activation("stem_feat_stem")
            )
            pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
                get_activation("stem_feat_stage_0")
            )
            pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
                get_activation("stem_feat_stage_1")
            )
            pretrained.model.patch_embed.backbone.stages[2].register_forward_hook(
                get_activation("stem_feat_stage_2")
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

    pretrained.act_postprocess1 = nn.Sequential(
        nn.Identity(), nn.Identity(), nn.Identity()
    )
    pretrained.act_postprocess2 = nn.Sequential(
        nn.Identity(), nn.Identity(), nn.Identity()
    )
    pretrained.act_postprocess3 = nn.Sequential(
        nn.Identity(), nn.Identity(), nn.Identity()
    )
    pretrained.act_postprocess4 = nn.Sequential(
        nn.Identity(), nn.Identity(), nn.Identity()
    )


    # if use_vit_only == True:
    #     pretrained.act_postprocess1 = nn.Sequential(
    #         readout_oper[0],
    #         Transpose(1, 2),
    #         nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
    #         nn.Conv2d(
    #             in_channels=vit_features,
    #             out_channels=features[0],
    #             kernel_size=1,
    #             stride=1,
    #             padding=0,
    #         ),
    #         nn.ConvTranspose2d(
    #             in_channels=features[0],
    #             out_channels=features[0],
    #             kernel_size=4,
    #             stride=4,
    #             padding=0,
    #             bias=True,
    #             dilation=1,
    #             groups=1,
    #         ),
    #     )

    #     pretrained.act_postprocess2 = nn.Sequential(
    #         readout_oper[1],
    #         Transpose(1, 2),
    #         nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
    #         nn.Conv2d(
    #             in_channels=vit_features,
    #             out_channels=features[1],
    #             kernel_size=1,
    #             stride=1,
    #             padding=0,
    #         ),
    #         nn.ConvTranspose2d(
    #             in_channels=features[1],
    #             out_channels=features[1],
    #             kernel_size=2,
    #             stride=2,
    #             padding=0,
    #             bias=True,
    #             dilation=1,
    #             groups=1,
    #         ),
    #     )
    # else:
    #     pretrained.act_postprocess1 = nn.Sequential(
    #         nn.Identity(), nn.Identity(), nn.Identity()
    #     )
    #     pretrained.act_postprocess2 = nn.Sequential(
    #         nn.Identity(), nn.Identity(), nn.Identity()
    #     )
    
    # pretrained.act_postprocess3 = nn.Sequential(
    #     readout_oper[2],
    #     Transpose(1, 2),
    #     nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
    #     nn.Conv2d(
    #         in_channels=vit_features,
    #         out_channels=features[2],
    #         kernel_size=1,
    #         stride=1,
    #         padding=0,
    #     ),
    # )

    # pretrained.act_postprocess4 = nn.Sequential(
    #     readout_oper[3],
    #     Transpose(1, 2),
    #     nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
    #     nn.Conv2d(
    #         in_channels=vit_features,
    #         out_channels=features[3],
    #         kernel_size=1,
    #         stride=1,
    #         padding=0,
    #     ),
    #     nn.Conv2d(
    #         in_channels=features[3],
    #         out_channels=features[3],
    #         kernel_size=3,
    #         stride=2,
    #         padding=1,
    #     ),
    # )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [opt.cfg.MODEL_BRDF.DPT_baseline.patch_size] * 2

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex_CAv2, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitb_rn50_384_CAv2(
    opt, 
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
):
    print('========= [_make_pretrained_vitb_rn50_384] pretrained', pretrained)

    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)

    model_tmp = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)
    if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_type == 'single':
        model.stem = model_tmp.patch_embed.backbone.stem
    elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_type == 'double':
        model.stem = nn.Sequential(
            model_tmp.patch_embed.backbone.stem, 
            model_tmp.patch_embed.backbone.stages[0]
        )
    elif opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_type == 'full':
        if not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_shared_stem:
            model.stem = model_tmp.patch_embed.backbone
        else:
            model.stem = nn.Module()

        backbone_dims = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_full.backbone_dims
        feat_proj_dims = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_full.proj_extra_dims

        proj_extra_if_inst_norm = opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_full.proj_extra_if_inst_norm

        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_full.proj_extra_if_simple:
            model.stem.proj_extra = nn.Conv2d(backbone_dims, feat_proj_dims, kernel_size=1, stride=1)
        else:
            model.stem.proj_extra = nn.Sequential(
                nn.Conv2d(backbone_dims, backbone_dims//2, kernel_size=1, stride=1), 
                nn.InstanceNorm2d(backbone_dims//2) if proj_extra_if_inst_norm else nn.Identity(), # to be consistent with self.ff_conv in Yogo-Transformer
                nn.ReLU(True),
                nn.Conv2d(backbone_dims//2, feat_proj_dims, kernel_size=1, stride=1), 
                nn.InstanceNorm2d(feat_proj_dims) if proj_extra_if_inst_norm else nn.Identity(),
                nn.ReLU(True),
                nn.Conv2d(feat_proj_dims, feat_proj_dims, kernel_size=1, stride=1)
            )

    else:
        assert False

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_vit_b_rn50_backbone_CAv2(
        opt, 
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )