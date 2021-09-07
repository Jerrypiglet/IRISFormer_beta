import time
from collections import OrderedDict

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from modules.functional.backend import _backend
# from modules.functional.sampling import gather, furthest_point_sample
# import modules.functional as mf

class LayerNormLastTwo(nn.Module):
    def __init__(self, dim):
        super(LayerNormLastTwo, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, A):
        return torch.transpose(self.ln(torch.transpose(A, -1, -2)), -1, -2)

def conv1x1_1d(inplanes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv1d(inplanes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)

def conv1x1(inplanes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv1d(inplanes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)

class MatMul(nn.Module):
    """A wrapper class such that we can count the FLOPs of matmul
    """
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)

class Transformer(nn.Module):
    def __init__(self, token_c, t_layer=1, head=2, kqv_groups=1,
                 norm_layer_1d=nn.Identity):
        super(Transformer, self).__init__()

        self.k_conv = nn.ModuleList()
        self.q_conv = nn.ModuleList()
        self.v_conv = nn.ModuleList()
        self.kqv_bn = nn.ModuleList()
        self.kq_matmul = nn.ModuleList()
        self.kqv_matmul = nn.ModuleList()
        self.ff_conv = nn.ModuleList()
        for _ in range(t_layer):
            self.k_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c // 2, groups=kqv_groups),
                norm_layer_1d(token_c // 2)
            ))
            self.q_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c // 2, groups=kqv_groups),
                norm_layer_1d(token_c // 2)
            ))
            self.v_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c, groups=kqv_groups),
                norm_layer_1d(token_c)
            ))
            self.kq_matmul.append(MatMul())
            self.kqv_matmul.append(MatMul())
            self.kqv_bn.append(norm_layer_1d(token_c))
            # zero-init
            #nn.init.constant_(self.kqv_bn[-1].weight, 0)
            self.ff_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c * 2),
                norm_layer_1d(token_c * 2),
                nn.ReLU(inplace=True),
                conv1x1_1d(token_c * 2, token_c),
                norm_layer_1d(token_c),
            ))
            # initialize the bn weight to zero to improves the training
            # stability.
            #nn.init.constant_(self.ff_conv[-1][1].weight, 1)

        self.token_c = token_c
        self.t_layer = t_layer
        self.head = head

    def forward(self, x):
        N = x.shape[0]
        for _idx in range(self.t_layer):
            k = self.k_conv[_idx](x).view(
                N, self.head, self.token_c // 2 // self.head, -1)
            q = self.q_conv[_idx](x).view(
                N, self.head, self.token_c // 2 // self.head, -1)
            v = self.v_conv[_idx](x).view(
                N, self.head, self.token_c // self.head, -1)
            # N, h, L, C/h * N, h, C/h, L -> N, h, L, L
            kq = self.kq_matmul[_idx](k.permute(0, 1, 3, 2), q)
            # N, h, L, L
            kq = F.softmax(kq / np.sqrt(self.token_c / 2 / self.head), dim=2)
            # N, h, C/h, L * N, h, L, L -> N, h, C/h, L
            kqv = self.kqv_matmul[_idx](v, kq).view(N, self.token_c, -1)
            kqv = self.kqv_bn[_idx](kqv)
            x = x + kqv
            x = x + self.ff_conv[_idx](x)

        return x

class Projector(nn.Module):
    def __init__(self, opt, token_c, output_c, head=2, min_group_planes=64,
                 norm_layer_1d=nn.Identity, ca_proj_method='full'):
        super(Projector, self).__init__()

        self.opt = opt
        self.ca_proj_method = ca_proj_method

        if token_c != output_c:
            self.proj_value_conv = nn.Sequential(
                conv1x1_1d(token_c, output_c),
                norm_layer_1d(output_c))
        else:
            self.proj_value_conv = nn.Identity()

        # self.proj_value_conv = nn.Sequential(
        #     conv1x1_1d(token_c, output_c),
        #     norm_layer_1d(output_c)
        # )

        self.proj_key_conv = nn.Sequential(
            conv1x1_1d(token_c, output_c),
            norm_layer_1d(output_c)
        )
        self.proj_query_conv = nn.Sequential(
            conv1x1_1d(output_c, output_c),
            norm_layer_1d(output_c)
        )
        self.proj_kq_matmul = MatMul()
        self.proj_matmul = MatMul()
        self.proj_bn = norm_layer_1d(output_c)
        # zero-init
        #nn.init.constant_(self.proj_bn.weight, 1)

        # if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_proj_method != 'none':
        self.ff_conv = nn.Sequential(
            conv1x1_1d(output_c, 2 * output_c),
            norm_layer_1d(2 * output_c),
            nn.ReLU(inplace=True),
            conv1x1_1d(2 * output_c, output_c),
            norm_layer_1d(output_c)
            )

        self.head = head

        # self.norm_out = norm_layer_1d(output_c)


    def forward(self, x, x_t, proj_coef_in=None, tokens_mask=None, im_mask=None):
        # print(x.shape, x_t.shape) # torch.Size([1, 512, 5120]) torch.Size([1, 768, 320])
        # print('----', x_t[0, :5, 0])
        # print('====', x_t[0, :5, -1])

        N, _, L = x_t.shape
        h = self.head
        # -> N, h, C/h, L
        proj_v = self.proj_value_conv(x_t).view(N, h, -1, L)
        # -> N, h, C/h, L
        # print(x_t.shape)
        proj_k = self.proj_key_conv(x_t).view(N, h, -1, L)
        proj_q = self.proj_query_conv(x)
        N, C, _ = proj_q.shape
        # -> N, h, HW, c/H
        proj_q = proj_q.view(N, h, C // h, -1).permute(0, 1, 3, 2)
        # N, h, HW, C/h * N, h, C/h, L -> N, h, HW, L
        # print(proj_q.shape, proj_k.shape, self.proj_kq_matmul(proj_q, proj_k).shape) # SSN: torch.Size([1, 2, 5120, 128]) torch.Size([1, 2, 128, 80]) torch.Size([1, 2, 5120, 80])
        proj_coef = self.proj_kq_matmul(proj_q, proj_k) / np.sqrt(C / h)
        # print('---->', proj_coef[0, 0, :5, 0])
        # print('====>', proj_coef[0, 0, :5, -1])
        if im_mask is not None:
            proj_coef = proj_coef * im_mask.unsqueeze(-1)

        if tokens_mask is not None:
            # print(tokens_mask.shape, proj_coef.shape) # torch.Size([1, 320]) torch.Size([1, 2, 5120, 320])
            tokens_mask_dim4 = tokens_mask.unsqueeze(1).unsqueeze(1)
            output_dict = {'proj_coef': proj_coef.clone() * tokens_mask_dim4}
        else:
            output_dict = {'proj_coef': proj_coef.clone()}

        if tokens_mask is not None:
            # print(tokens_mask.shape, proj_coef.shape) # torch.Size([1, 320]) torch.Size([1, 2, 5120, 320])
            # tokens_mask_dim4 = tokens_mask.unsqueeze(1).unsqueeze(1)
            # proj_coef = proj_coef * tokens_mask_dim4 + torch.ones_like(proj_coef) * (-1e6) * (1. - tokens_mask_dim4) # masked softmax
            proj_coef = torch.exp(proj_coef) / ((torch.exp(proj_coef) * tokens_mask_dim4).sum(-1, keepdims=True) + 1e-6)
            proj_coef = proj_coef * tokens_mask_dim4
            # print(tokens_mask[0])
            # print(proj_coef[0].sum(0).sum(0)) # should be ones and zeros
            # print(proj_coef[0].sum(-1)) # should be all ones
        else:
            proj_coef = F.softmax(proj_coef, dim=3)

        
        # print(proj_coef.shape) # torch.Size([-1, 2 (head num), 5120, 320]), torch.Size([-1, 2 (head num), 1280, 320]), torch.Size([-1, 2 (head num), 320, 320]), torch.Size([-1, 2 (head num), 80, 320]), 
        # print('---proj_coef', proj_coef.shape)
        
        if self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_use_Q_as_proj_coef:
            assert False, 'disabled for now'
            assert proj_coef_in is not None
            # print('---proj_coef_in', proj_coef_in.shape)

            proj_coef_in = proj_coef_in.flatten(2).transpose(-1, -2).unsqueeze(1).repeat(1, self.head, 1, 1)
            # proj_coef_in = F.softmax(proj_coef_in / np.sqrt(C / h), dim=3)
            proj_coef_in = F.softmax(proj_coef_in, dim=3)
            
            # print('---proj_coef_in', proj_coef_in.shape)
            assert proj_coef_in.shape==proj_coef.shape
            # proj_coef = proj_coef * 0. + proj_coef_in
            proj_coef = proj_coef * proj_coef_in

        # N, h, C/h, L * N, h, L, HW -> N, h, C/h, HW
        # print(tokens_mask.shape, proj_v.shape, proj_coef.shape) # torch.Size([1, 80]) torch.Size([1, 2, 256, 80]) torch.Size([1, 2, 5120, 80])
        # print(tokens_mask)
        if tokens_mask is not None:
            proj_v = proj_v * tokens_mask_dim4
            proj_coef = proj_coef * tokens_mask_dim4
            # print(proj_coef.shape, tokens_mask)
        x_p = self.proj_matmul(proj_v, proj_coef.permute(0, 1, 3, 2))
        # -> N, C, H, W
        _, _, S = x.shape
        x_p = self.proj_bn(x_p.view(N, -1, S))
        # print('-=-=', x.shape, x_p.shape) # -=-= torch.Size([1, 768, 5120]) torch.Size([1, 768, 5120])

        # print(self.ca_proj_method)
        if self.ca_proj_method == 'residual':
            # print('---',torch.mean(x), torch.median(x), torch.max(x), torch.min(x))
            # print('---+',torch.mean(x_p), torch.median(x_p), torch.max(x_p), torch.min(x_p))
            # print(self.ff_conv)
            # a = self.ff_conv[0](x + x_p)
            # b = self.ff_conv[1](a)
            # print('---+a',torch.mean(a), torch.median(a), torch.max(a), torch.min(a))
            # print('---+b',torch.mean(b), torch.median(b), torch.max(b), torch.min(b))
            x = self.ff_conv(x + x_p)
            # print(x.shape, proj_coef.shape) # torch.Size([1, 768, 5120]) torch.Size([1, 2, 5120, 320])
            # print('--->',torch.mean(x), torch.median(x), torch.max(x), torch.min(x))
            # x = self.norm_out(x)
            # print('--->>>>>',torch.mean(x), torch.median(x), torch.max(x), torch.min(x))
            output_dict['x'] = x
        elif self.ca_proj_method == 'full':
            x = x + self.ff_conv(x + x_p)
            output_dict['x'] = x
        elif self.ca_proj_method == 'concat':
            x = torch.cat([x, self.ff_conv(x + x_p)], 1)
            output_dict['x'] = x
        elif self.ca_proj_method == 'none':
            output_dict['x'] = self.ff_conv(x_p)
        else:
            assert False

        return output_dict

class CrossAttention(nn.Module):
    def __init__(self, opt, token_c, input_dims, output_dims,
                 head=2, min_group_planes=1, norm_layer_1d=nn.Identity,
                 ca_proj_method='full', 
                 **kwargs):
        super(CrossAttention, self).__init__()

        self.opt = opt

        self.token_c = token_c
        self.input_dims = input_dims

        # if input_dims == output_dims:
        #     self.feature_block = nn.Identity()  
        # else:
        #     self.feature_block = nn.Sequential(
        #             conv1x1_1d(input_dims, output_dims),
        #             norm_layer_1d(output_dims)
        #             )

        # self.feature_block = nn.Sequential(
        #     conv1x1_1d(input_dims, output_dims),
        #     norm_layer_1d(output_dims)
        #     )

        self.projectors = Projector(
                    self.opt, 
                    token_c, output_dims, head=head,
                    min_group_planes=min_group_planes,
                    norm_layer_1d=norm_layer_1d, 
                    ca_proj_method=ca_proj_method)


    def forward(self, in_feature, in_tokens, im_feat_scale_factor=1., proj_coef_in=None):
        # pass
        batch_size, im_feat_dim, im_h, im_w = in_feature.shape[:4]
        assert self.token_c == in_tokens.shape[1]
        assert self.input_dims==im_feat_dim
        if im_feat_scale_factor != 1.:
            im_feat_resized = F.interpolate(in_feature, scale_factor=im_feat_scale_factor, mode='bilinear') # torch.Size([1, 1344, 16, 20])
        else:
            im_feat_resized = in_feature

        im_feat_flattened = im_feat_resized.view(batch_size, im_feat_dim, -1)

        output_dict = self.projectors(
            # self.feature_block(im_feat_flattened), in_tokens, proj_coef_in=proj_coef_in
            im_feat_flattened, in_tokens, proj_coef_in=proj_coef_in
            ) 
        out_feature = output_dict['x']
        proj_coef = output_dict['proj_coef']
        # print(proj_coef.shape) # torch.Size([1, 2, 5120, 320])

        out_feature = out_feature.view(batch_size, self.token_c, int(im_h*im_feat_scale_factor), int(im_w*im_feat_scale_factor))

        return out_feature, proj_coef


class CrossAttention_CAv2(nn.Module):
    def __init__(self, opt, token_c, input_dims, output_dims,
                 head=2, min_group_planes=1, norm_layer_1d=nn.Identity,
                 ca_proj_method='full', 
                 **kwargs):
        super(CrossAttention_CAv2, self).__init__()

        self.opt = opt

        self.token_c = token_c
        self.input_dims = input_dims
        self.output_dims = output_dims

        if input_dims == output_dims:
            self.feature_block = nn.Identity()  
        else:
            self.feature_block = nn.Sequential(
                conv1x1_1d(input_dims, output_dims),
                norm_layer_1d(output_dims)
                )

        self.projectors = Projector(
                    self.opt, 
                    token_c, output_dims, head=head,
                    min_group_planes=min_group_planes,
                    norm_layer_1d=norm_layer_1d, 
                    ca_proj_method=ca_proj_method)


    def forward(self, in_feature, in_tokens, im_feat_scale_factor=1., proj_coef_in=None, if_in_feature_flattened=False, tokens_mask=None, im_mask=None, im_mask_scale_factor=1., ):
        # print(self.token_c, in_tokens.shape, self.input_dims, in_feature.shape)
        assert self.token_c == in_tokens.shape[1]
        if if_in_feature_flattened:
            im_feat_flattened = in_feature
            assert len(im_feat_flattened.shape) == 3
            assert im_feat_scale_factor == 1.
            assert self.input_dims == im_feat_flattened.shape[1]
        else:
            batch_size, im_feat_dim, im_h, im_w = in_feature.shape[:4]
            assert self.input_dims==im_feat_dim
            if im_feat_scale_factor != 1.:
                im_feat_resized = F.interpolate(in_feature, scale_factor=im_feat_scale_factor, mode='bilinear') # torch.Size([1, 1344, 16, 20])
            else:
                im_feat_resized = in_feature

            im_feat_flattened = im_feat_resized.flatten(2)

            if im_mask is not None:
                if im_mask_scale_factor != 1.:
                    im_mask_resized = F.interpolate(im_mask.unsqueeze(1), scale_factor=im_mask_scale_factor, mode='bilinear') # torch.Size([1, 1, 16, 20])
                else:
                    im_mask_resized = im_mask
                im_mask_flattened = im_mask_resized.flatten(2)
                assert im_mask_flattened.shape[2:]==im_feat_flattened.shape[2:]

        output_dict = self.projectors(
            self.feature_block(im_feat_flattened), 
            in_tokens, 
            proj_coef_in=proj_coef_in, 
            tokens_mask=tokens_mask,
            im_mask=im_mask_flattened
        ) 
        out_feature = output_dict['x']
        proj_coef = output_dict['proj_coef']
        # print(proj_coef.shape) # torch.Size([1, 2, 5120, 320])

        # print(in_feature.shape, out_feature.shape)
        if not if_in_feature_flattened:
            out_feature = out_feature.view(batch_size, self.output_dims, int(im_h*im_feat_scale_factor), int(im_w*im_feat_scale_factor))

        return out_feature, proj_coef

    

class RIM(nn.Module):
    def __init__(self, token_c, input_dims, output_dims,
                 head=2, min_group_planes=1, norm_layer_1d=nn.Identity,
                 **kwargs):
        super(RIM, self).__init__()

        self.transformer = Transformer(
            token_c, norm_layer_1d=norm_layer_1d, head=head,
            **kwargs)

        if input_dims == output_dims:
            self.feature_block = nn.Identity()  
        else:
            self.feature_block = nn.Sequential(
                    conv1x1_1d(input_dims, output_dims),
                    norm_layer_1d(output_dims)
                    )

        self.projectors = Projector(
                    token_c, output_dims, head=head,
                    min_group_planes=min_group_planes,
                    norm_layer_1d=norm_layer_1d)
        
        self.dynamic_f = nn.Sequential(
            conv1x1_1d(input_dims, token_c),
            norm_layer_1d(token_c),
            nn.ReLU(inplace=True),
            conv1x1_1d(token_c, token_c),
            norm_layer_1d(token_c)
            )

    def forward(self, in_feature, in_tokens, knn_idx):
        #in_feature: B, N, C
        #in_coords: B, N, 3
        B, L, K = knn_idx.shape
        B, C, N = in_feature.shape

        gather_fts = gather(
                in_feature, knn_idx.view(B, -1)
                ).view(B, -1, L, K)
       
        tokens = self.dynamic_f(gather_fts.max(dim=3)[0])
  
        t_c = tokens.shape[1]
        
        if in_tokens is not None:
            tokens += in_tokens
     
        tokens = self.transformer(tokens)

        out_feature = self.projectors(
                    self.feature_block(in_feature), tokens
                    ) 

        return out_feature, tokens

class RIM_ResidualBlock(nn.Module):
    def __init__(self, inc, outc, token_c, norm_layer_1d):
        super(RIM_ResidualBlock, self).__init__()
        if inc != outc:
            self.res_connect = nn.Sequential(
                    nn.Conv1d(inc, outc, 1),
                    norm_layer_1d(outc),
                    )
        else:
            self.res_connect = nn.Identity()
        self.vt1 = RIM(
               token_c, inc, inc, norm_layer_1d=norm_layer_1d)
        self.vt2 = RIM(
               token_c, inc, outc, norm_layer_1d=norm_layer_1d)

    def forward(self, inputs):
        in_feature, tokens, knn_idx = inputs    
        out, tokens = self.vt1(in_feature, tokens, knn_idx)
        out, tokens = self.vt2(out, tokens, knn_idx)

        return out, tokens

class YOGO(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs['width_r']
        cs = [32, 64, 128, 256, 256]
        cs = [int(cr * x) for x in cs]

        self.token_l = kwargs['token_l']
        self.token_s = kwargs['token_s']
        self.token_c = kwargs['token_c']

        self.group_ = kwargs['group']
        self.ball_r = kwargs['ball_r']

        norm_layer = kwargs['norm']
        
        self.stem = nn.Sequential(
            conv1x1_1d(22, cs[0]),
            norm_layer(cs[0]),
            nn.ReLU(inplace=True),
            conv1x1_1d(cs[0], cs[0]),
            norm_layer(cs[0])
        )

        self.stage1 = nn.Sequential(
            RIM_ResidualBlock(cs[0], cs[1], token_c=self.token_c, norm_layer_1d=norm_layer),
        )

        self.stage2 = nn.Sequential(
            RIM_ResidualBlock(cs[1], cs[2], token_c=self.token_c, norm_layer_1d=norm_layer),
        )

        self.stage3 = nn.Sequential(
            RIM_ResidualBlock(cs[2], cs[3], token_c=self.token_c, norm_layer_1d=norm_layer),
        )

        self.stage4 = nn.Sequential(
            RIM_ResidualBlock(cs[3], cs[4], token_c=self.token_c, norm_layer_1d=norm_layer),
        ) 

        self.classifier = nn.Sequential(
            conv1x1_1d(cs[4], cs[4]),
            norm_layer(cs[4]),
            nn.ReLU(inplace=True),
            conv1x1_1d(cs[4], kwargs['num_classes']),
            )

    def forward(self, x):

        coords = x[:, :3, :]

        one_hot_vectors = x[:, -16:, :]

        B, _, N = x.shape

        feature_stem = self.stem(x)
        if self.training: 
            token_l = int(
                np.random.randint(self.token_l-8, self.token_l+8))
            center_pts = furthest_point_sample(
                coords, token_l)
        else:
            center_pts = furthest_point_sample(
                coords, self.token_l)

        if self.group_ == 'ball_query':
            knn_idx = mf.ball_query(
                center_pts, coords, self.ball_r, self.token_s
                )
        else:
            knn_idx = knn_search(coords, center_pts, self.token_s)       
            knn_idx = torch.from_numpy(knn_idx).cuda()
        
        feature1, tokens = self.stage1((feature_stem, None, knn_idx))
              
        feature2, tokens = self.stage2((feature1, tokens, knn_idx))

        feature3, tokens = self.stage3((feature2, tokens, knn_idx))

        feature4, tokens = self.stage4((feature3, tokens, knn_idx))

        out = self.classifier(feature4)
        return out


