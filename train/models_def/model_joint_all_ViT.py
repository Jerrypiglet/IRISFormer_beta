from math import e
import torch
import torch.nn as nn

from models_def.model_matseg import Baseline
from utils.utils_misc import *
# import pac
from utils.utils_training import freeze_bn_in_module, unfreeze_bn_in_module
import torch.nn.functional as F
from torchvision.models import resnet

from models_def.model_matseg import logit_embedding_to_instance

import models_def.models_brdf as models_brdf # basic model
import models_def.models_brdf_GMM_feat_transform as models_brdf_GMM_feat_transform
# import models_def.models_brdf_pac_pool as models_brdf_pac_pool
# import models_def.models_brdf_pac_conv as models_brdf_pac_conv
# import models_def.models_brdf_safenet as models_brdf_safenet
import models_def.models_light as models_light 
import models_def.models_layout_emitter as models_layout_emitter
import models_def.models_object_detection as models_object_detection
import models_def.models_mesh_reconstruction as models_mesh_reconstruction
import models_def.models_layout_emitter_lightAccu as models_layout_emitter_lightAccu
import models_def.models_layout_emitter_lightAccuScatter as models_layout_emitter_lightAccuScatter
import models_def.model_matcls as model_matcls
# import models_def.model_nvidia.AppGMM as AppGMM
import models_def.model_nvidia.AppGMM_singleFrame as AppGMM
import models_def.model_nvidia.ssn.ssn as ssn
import models_def.models_swin as models_swin
from models_def.models_swin import get_LightNet_Swin

from utils.utils_scannet import convert_IntM_from_OR, CamIntrinsic_to_cuda

from SimpleLayout.SimpleSceneTorchBatch import SimpleSceneTorchBatch
from utils.utils_total3D.utils_OR_layout import get_layout_bdb_sunrgbd
from utils.utils_total3D.utils_OR_cam import get_rotation_matix_result

from models_def.model_dpt.models import DPTBRDFModel, get_BRDFNet_DPT, get_LightNet_DPT
from models_def.model_dpt.models_CAv2 import DPTBRDFModel_CAv2, get_BRDFNet_DPT_CAv2
from models_def.model_dpt.models_SSN import DPTBRDFModel_SSN
from models_def.model_dpt.models_SSN_yogoUnet_N_layers import DPTBRDFModel_SSN_yogoUnet_N_layers
from models_def.model_dpt.transforms import Resize as dpt_Resize
from models_def.model_dpt.transforms import NormalizeImage as dpt_NormalizeImage
from models_def.model_dpt.transforms import PrepareForNet as dpt_PrepareForNet

# from models_def.model_dpt.models_ViT import get_LayoutNet_ViT
# from models_def.model_dpt.blocks_ViT import forward_vit_ViT
# from torchvision.transforms import Compose
# import cv2
# import time

from icecream import ic

# from models_def.model_dpt.blocks import forward_vit
import torch.utils.checkpoint as cp

from models_def.model_dpt.models_ALL_ViT_DPT import ModelAll_ViT

import models_def.BilateralLayer as bs

class Model_Joint_ViT(nn.Module):
    def __init__(self, opt, logger):
        super(Model_Joint_ViT, self).__init__()
        self.opt = opt
        self.cfg = opt.cfg
        self.logger = logger

        assert self.opt.cfg.MODEL_ALL.ViT_baseline.enable

        self.modalities = self.opt.cfg.MODEL_ALL.enable_list
        if 'li' in self.modalities:
            self.modalities.remove('li')
            self.modalities += ['axis', 'lamb', 'weight']

        self.modalities_stage0 = list(set(self.modalities) & set(['al', 'no', 'de', 'ro', 'lo']))
        self.modalities_stage0.sort() # make sure albedo comest first for BS purpose
        self.modalities_stage1 = list(set(self.modalities) & set(['axis', 'lamb', 'weight']))
        self.if_BRDF = self.modalities_stage0 != []
        self.if_Light = self.modalities_stage1 != []
        self.load_brdf_gt = self.opt.cfg.DATA.load_brdf_gt
        self.use_GT_brdf = self.opt.cfg.MODEL_LIGHT.use_GT_brdf or self.opt.cfg.MODEL_LIGHT.use_offline_brdf

        print('====self.modalities', self.modalities)
        self.MODEL_ALL = ModelAll_ViT(
            opt=opt, 
            modalities=self.modalities.copy(), 
            backbone="vitb_rn50_384_N_layers", 
            N_layers_encoder_stage0 = self.cfg.MODEL_ALL.ViT_baseline.N_layers_encoder_stage0, 
            N_layers_decoder_stage0 = self.cfg.MODEL_ALL.ViT_baseline.N_layers_decoder_stage0, 
            N_layers_encoder_stage1 = self.cfg.MODEL_ALL.ViT_baseline.N_layers_encoder_stage1, 
            N_layers_decoder_stage1 = self.cfg.MODEL_ALL.ViT_baseline.N_layers_decoder_stage1
        )
        self.forward_LightNet_func = self.MODEL_ALL.forward_light

        if self.if_Light:
            self.non_learnable_layers = {}
            self.non_learnable_layers['renderLayer'] = models_light.renderingLayer(isCuda = opt.if_cuda, 
                imWidth=opt.cfg.MODEL_LIGHT.envCol, imHeight=opt.cfg.MODEL_LIGHT.envRow, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight)
            self.non_learnable_layers['output2env'] = models_light.output2env(isCuda = opt.if_cuda, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight, SGNum = opt.cfg.MODEL_LIGHT.SGNum )

            if self.cfg.MODEL_LIGHT.freeze_BRDF_Net and self.if_BRDF:
                self.freeze_BRDF()

            if self.cfg.MODEL_ALL.ViT_baseline.if_UNet_lighting:
                self.MODEL_ALL._.shared_encoder_stage1 = nn.Identity()
                self.MODEL_ALL._.axis = nn.Identity()
                self.MODEL_ALL._.lamb = nn.Identity()
                self.MODEL_ALL._.weight = nn.Identity()

                self.LIGHT_Net = nn.ModuleDict({})
                self.LIGHT_Net.update({'lightEncoder':  models_light.encoderLight(cascadeLevel = opt.cascadeLevel, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
                self.LIGHT_Net.update({'axisDecoder':  models_light.decoderLight(mode=0, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
                self.LIGHT_Net.update({'lambDecoder':  models_light.decoderLight(mode = 1, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
                self.LIGHT_Net.update({'weightDecoder':  models_light.decoderLight(mode = 2, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})

                self.forward_LightNet_func = self.forward_LightNet_UNet_func

        if self.cfg.MODEL_BRDF.if_bilateral:
            # assert self.cfg.DEBUG.if_test_real
            self.BRDF_Net = nn.ModuleDict({})
            if 'al' in self.modalities_stage0:
                self.BRDF_Net.update({'albedoBs': bs.BilateralLayer(mode = 0).eval() })
                for param in self.BRDF_Net['albedoBs'].parameters():
                    param.requires_grad = False
            if 'no' in self.modalities_stage0:
                self.BRDF_Net.update({'normalBs': bs.BilateralLayer(mode = 1).eval() })
                for param in self.BRDF_Net['normalBs'].parameters():
                    param.requires_grad = False
            if 'ro' in self.modalities_stage0:
                self.BRDF_Net.update({'roughBs': bs.BilateralLayer(mode = 2).eval() })
                for param in self.BRDF_Net['roughBs'].parameters():
                    param.requires_grad = False
            if 'de' in self.modalities_stage0:
                self.BRDF_Net.update({'depthBs': bs.BilateralLayer(mode = 4).eval() })
                for param in self.BRDF_Net['depthBs'].parameters():
                    param.requires_grad = False



        if self.cfg.MODEL_LIGHT.load_pretrained_MODEL_BRDF:
            self.load_pretrained_MODEL_BRDF(if_load_encoder=self.cfg.MODEL_BRDF.pretrained_if_load_encoder, if_load_decoder=self.cfg.MODEL_BRDF.pretrained_if_load_decoder, if_load_Bs=self.cfg.MODEL_BRDF.pretrained_if_load_Bs)

    def freeze_partsset_to_val(self):
        if self.cfg.MODEL_LIGHT.freeze_BRDF_Net and self.if_BRDF:
            self.freeze_BRDF()

    def forward(self, input_dict, if_has_gt_BRDF=True):
        # module_hooks_dict = {}
        # input_dict_extra = {}
        output_dict = {}

        # print(input_dict['input_batch_brdf'].shape)
        # input_dict_extra['shared_encoder_outputs'] = forward_vit_ViT(
        #     self.opt, self.opt.cfg.MODEL_ALL.ViT_baseline, self.MODEL_ALL.shared_encoder, 
        #     input_dict['input_batch_brdf'])
        if self.if_BRDF:
            # assert self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage0
            x_stage0 = input_dict['input_batch_brdf']
            # ic(x_stage0.shape)
            return_dict_brdf = self.forward_brdf(x_stage0, input_dict, if_has_gt_BRDF=if_has_gt_BRDF)
            output_dict.update(return_dict_brdf)
        else:
            return_dict_brdf = {}

        if self.if_Light:
            if self.opt.cfg.DEBUG.if_test_real:
                return_dict_light = self.forward_light_real(input_dict, return_dict_brdf)
            else:
                return_dict_light = self.forward_light(input_dict, return_dict_brdf)
            output_dict.update(return_dict_light)

        return output_dict


    def forward_brdf(self, x_stage0, input_dict, if_has_gt_BRDF=True):
        output_dict_model_all = self.MODEL_ALL.forward_brdf(x_stage0)

        if_has_gt_BRDF = if_has_gt_BRDF and (not self.opt.cfg.DATASET.if_no_gt_BRDF) and self.load_brdf_gt

        return_dict = {}

        # im_h_resized_to_batch, im_w_resized_to_batch = input_dict['im_h_resized_to'], input_dict['im_w_resized_to']

        for modality in self.modalities_stage0:
            # vit_out = self.MODEL_ALL[modality].forward(None, input_dict_extra=input_dict_extra)
            vit_out = output_dict_model_all[modality]

            if modality == 'al':
                albedoPred = 0.5 * (vit_out + 1)
                albedoPred = torch.clamp(albedoPred, 0, 1)
                return_dict.update({'albedoPred': albedoPred})
                if if_has_gt_BRDF and 'al' in self.opt.cfg.DATA.data_read_list:
                    input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
                    albedoPred_aligned = models_brdf.LSregress(
                        albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred),
                        input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), 
                        albedoPred)
                    albedoPred_aligned = torch.clamp(albedoPred_aligned, 0, 1)
                    return_dict.update({'albedoPred_aligned': albedoPred_aligned})

                if self.cfg.MODEL_BRDF.if_bilateral:
                    # assert self.cfg.DEBUG.if_test_real
                    albedoBsPred, albedoConf = self.BRDF_Net['albedoBs'](input_dict['imBatch'], albedoPred.detach(), albedoPred )
                    return_dict.update({'albedoBsPred': albedoBsPred})
                    # if if_has_gt_BRDF and 'al' in self.opt.cfg.DATA.data_read_list:
                    #     albedoBsPred = models_brdf.LSregress(
                    #         albedoBsPred * input_dict['segBRDFBatch'].expand_as(albedoBsPred ),
                    #         input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), 
                    #         albedoBsPred )
                    # albedoBsPred = torch.clamp(albedoBsPred, 0, 1 )
                    # if if_has_gt_BRDF and 'al' in self.opt.cfg.DATA.data_read_list:
                    #     albedoBsPred = input_dict['segBRDFBatch'] * albedoBsPred
                    # return_dict.update({'albedoBsPred': albedoBsPred, 'albedoConf': albedoConf})

            elif modality == 'de':
                if self.opt.cfg.MODEL_BRDF.loss.depth.if_use_Zhengqin_loss:
                    depthPred = vit_out
                elif self.opt.cfg.MODEL_BRDF.loss.depth.if_use_midas_loss:
                    return_dict.update({'depthInvPred': vit_out})
                    depthPred = 1. / (vit_out + 1e-8)
                else:
                    assert False
                return_dict.update({'depthPred': depthPred})
                # print(if_has_gt_BRDF)
                # print((not self.opt.cfg.DATASET.if_no_gt_BRDF), self.load_brdf_gt)
                if if_has_gt_BRDF and 'de' in self.opt.cfg.DATA.data_read_list:
                    if 'segAllBatch' in input_dict:
                        depthPred_aligned = models_brdf.LSregress(
                            depthPred.detach() *  input_dict['segAllBatch'].expand_as(depthPred),
                            input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), 
                            depthPred)
                    elif 'segDepthBatch' in input_dict:
                        depthPred_aligned = models_brdf.LSregress(
                            depthPred.detach() *  input_dict['segDepthBatch'].expand_as(depthPred),
                            input_dict['depthBatch'] * input_dict['segDepthBatch'].expand_as(input_dict['depthBatch']), 
                            depthPred)
                    else:
                        assert False
                    return_dict.update({'depthPred_aligned': depthPred_aligned})

                if self.cfg.MODEL_BRDF.if_bilateral:
                    # assert self.cfg.DEBUG.if_test_real
                    depthBsPred, depthConf = self.BRDF_Net['depthBs'](input_dict['imBatch'], return_dict['albedoPred'].detach(), depthPred )
                    return_dict.update({'depthBsPred': depthBsPred})

            elif modality == 'ro':
                roughPred = vit_out
                return_dict.update({'roughPred': roughPred})
                if self.cfg.MODEL_BRDF.if_bilateral:
                    # assert self.cfg.DEBUG.if_test_real
                    # print(torch.max(roughPred), torch.min(roughPred), torch.median(roughPred), roughPred.shape)
                    # print(torch.max(roughPred[:, :, :213, :]), torch.min(roughPred[:, :, :213, :]), torch.median(roughPred[:, :, :213, :]), roughPred[:, :, :213, :].shape)
                    # print(torch.max(return_dict['albedoPred']), torch.min(return_dict['albedoPred']), torch.median(return_dict['albedoPred']), return_dict['albedoPred'].shape)
                    # roughBsPred, roughConf = self.BRDF_Net['roughBs'](input_dict['imBatch'][:, :, :213, :], return_dict['albedoPred'][:, :, :213, :].detach(), 0.5*(roughPred[:, :, :213, :]+1.) )
                    roughBsPred, roughConf = self.BRDF_Net['roughBs'](input_dict['imBatch'], return_dict['albedoPred'].detach(), 0.5*(roughPred+1.) )
                    roughBsPred = torch.clamp(2 * roughBsPred - 1, -1, 1)
                    # roughBsPred = roughPred[:, :, :213, :]
                    return_dict.update({'roughBsPred': roughBsPred})

            elif modality == 'no':
                normalPred = vit_out
                return_dict.update({'normalPred': normalPred})

            elif modality == 'lo':
                # return_dict.update({'layout_est_result': vit_out})
                return_dict.update(vit_out) # {'layout_est_result':...
            else:
                assert False, 'Unsupported modality: %s'%modality

        return return_dict


    def forward_light(self, input_dict, return_dict_brdf={}):
        im_h, im_w = self.cfg.DATA.im_height, self.cfg.DATA.im_width

        imBatch = input_dict['imBatch']
        segBRDFBatch = input_dict['segBRDFBatch']
        pad_mask = input_dict['pad_mask'].float().unsqueeze(1)

        # if not self.opt.cfg.MODEL_LIGHT.if_image_only_input:
        # Normalize Albedo and depth
        if 'al' in self.modalities_stage0:
            if self.cfg.MODEL_BRDF.if_bilateral:
                albedoInput = return_dict_brdf['albedoBsPred'].detach().clone()
            else:
                albedoInput = return_dict_brdf['albedoPred'].detach().clone() # always use non-aligned version as input!!!
        else:
            assert self.use_GT_brdf or self.opt.cfg.MODEL_LIGHT.if_image_only_input
            albedoInput = input_dict['albedoBatch'].detach().clone()

        if 'de' in self.modalities_stage0:
            if self.cfg.MODEL_BRDF.if_bilateral:
                depthInput = return_dict_brdf['depthBsPred'].detach().clone()
            else:
                depthInput = return_dict_brdf['depthPred'].detach().clone()
        else:
            assert self.use_GT_brdf or self.opt.cfg.MODEL_LIGHT.if_image_only_input
            depthInput = input_dict['depthBatch'].detach().clone()

        if 'no' in self.modalities_stage0:
            normalInput = return_dict_brdf['normalPred'].detach().clone()
        else:
            assert self.use_GT_brdf or self.opt.cfg.MODEL_LIGHT.if_image_only_input
            normalInput = input_dict['normalBatch'].detach().clone()

        if 'ro' in self.modalities_stage0:
            if self.cfg.MODEL_BRDF.if_bilateral:
                roughInput = return_dict_brdf['roughBsPred'].detach().clone()
            else:
                roughInput = return_dict_brdf['roughPred'].detach().clone()
        else:
            assert self.use_GT_brdf or self.opt.cfg.MODEL_LIGHT.if_image_only_input
            roughInput = input_dict['roughBatch'].detach().clone()


        albedoInput[:, :, :im_h, :im_w] = albedoInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(albedoInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0
        albedoInput = albedoInput * pad_mask
        # print(albedoInput.shape, pad_mask.shape)
        # albedoInput[:, :, im_h:, :] = 0.
        # albedoInput[:, :, :, im_w:] = 0.
        # albedoInput[:, :, im_h:, im_w:] = 0.

        # if self.opt.is_master:
        #     print('--depth before', torch.max(depthInput), torch.min(depthInput), torch.mean(depthInput), torch.median(depthInput))

        depthInput = torch.clamp(depthInput, 0., self.cfg.MODEL_LIGHT.depth_thres)
        depthInput[:, :, :im_h, :im_w] = depthInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(depthInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0
        depthInput = depthInput * pad_mask

        normalInput[:, :, :im_h, :im_w] =  0.5 * (normalInput[:, :, :im_h, :im_w] + 1)
        normalInput = normalInput * pad_mask

        roughInput[:, :, :im_h, :im_w] =  0.5 * (roughInput[:, :, :im_h, :im_w] + 1)
        roughInput = roughInput * pad_mask
        

        # if self.opt.is_master:
        #     print(torch.max(albedoInput), torch.min(albedoInput), torch.mean(albedoInput), torch.median(albedoInput))
        #     print(torch.max(depthInput), torch.min(depthInput), torch.mean(depthInput), torch.median(depthInput))
        #     print(torch.max(normalInput), torch.min(normalInput), torch.mean(normalInput), torch.median(normalInput))
        #     print(torch.max(roughInput), torch.min(roughInput), torch.mean(roughInput), torch.median(roughInput))
        
        if self.opt.cfg.MODEL_LIGHT.if_image_only_input:
            x_stage1 = imBatch.detach()
        else:
            x_stage1 = torch.cat([imBatch, albedoInput, normalInput, roughInput, depthInput ], dim=1 ).detach()

        # output_dict_model_all = self.MODEL_ALL.forward_light(x_stage1)
        # print(x_stage1.shape)
        # print(imBatch.shape, albedoInput.shape, normalInput.shape, roughInput.shape, depthInput.shape)
        output_dict_model_all = self.forward_LightNet_func(x_stage1)

        axisPred_ori, lambPred_ori, weightPred_ori = output_dict_model_all['axis'], output_dict_model_all['lamb'], output_dict_model_all['weight']
        if self.opt.cfg.MODEL_LIGHT.if_est_log_weight:
            weightPred_ori = torch.exp(weightPred_ori)

        if self.opt.is_master:
            print('weightPred_ori', torch.max(weightPred_ori), torch.min(weightPred_ori), torch.mean(weightPred_ori), torch.median(weightPred_ori))

        if self.cfg.DATA.if_pad_to_32x:
            axisPred_ori, lambPred_ori, weightPred_ori = axisPred_ori[:, :, :, :im_h//2, :im_w//2], lambPred_ori[:, :, :im_h//2, :im_w//2], weightPred_ori[:, :, :im_h//2, :im_w//2]
            imBatch = imBatch[:, :, :im_h, :im_w]
            segBRDFBatch = segBRDFBatch[:, :, :im_h, :im_w]

        imBatchSmall = F.adaptive_avg_pool2d(imBatch, (self.cfg.MODEL_LIGHT.envRow, self.cfg.MODEL_LIGHT.envCol) )
        segBRDFBatchSmall = F.interpolate(segBRDFBatch, scale_factor=0.5, mode="nearest")
        notDarkEnv = (torch.mean(torch.mean(torch.mean(input_dict['envmapsBatch'], 4), 4), 1, True ) > 0.001 ).float()
        segEnvBatch = (segBRDFBatchSmall * input_dict['envmapsIndBatch'].expand_as(segBRDFBatchSmall) ).unsqueeze(-1).unsqueeze(-1)
        segEnvBatch = segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)
        
        return_dict = {}

        envmapsPredImage, axisPred, lambPred, weightPred = self.non_learnable_layers['output2env'].output2env(axisPred_ori, lambPred_ori, weightPred_ori, if_postprocessing=not self.cfg.MODEL_LIGHT.use_GT_light_sg)

        pixelNum_recon = max( (torch.sum(segEnvBatch ).cpu().data).item(), 1e-5)
        if self.cfg.MODEL_LIGHT.use_GT_light_sg:
            envmapsPredScaledImage = envmapsPredImage * (input_dict['hdr_scaleBatch'].flatten().view(-1, 1, 1, 1, 1, 1))
            envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        elif self.cfg.MODEL_LIGHT.use_GT_light_envmap:
            envmapsPredScaledImage = envmapsPredImage # gt envmap already scaled in dataloader
            envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        elif self.cfg.MODEL_LIGHT.use_scale_aware_loss:
            envmapsPredScaledImage = envmapsPredImage # not aligning envmap
            envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        else: # scale-invariant
            if self.cfg.MODEL_LIGHT.if_align_log_envmap:
                # assert False, 'disabled'
                # envmapsPredScaledImage = models_brdf.LSregress(torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset).detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                #     torch.log(input_dict['envmapsBatch'] + self.cfg.MODEL_LIGHT.offset) * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
                #     if_clamp_coeff=False)
                envmapsPredScaledImage_offset_log_ = models_brdf.LSregress(torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset).detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                    torch.log(input_dict['envmapsBatch'] + self.cfg.MODEL_LIGHT.offset) * segEnvBatch.expand_as(input_dict['envmapsBatch']), torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset), 
                    if_clamp_coeff=self.cfg.MODEL_LIGHT.if_clamp_coeff)
                envmapsPredScaledImage = models_brdf.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                    input_dict['envmapsBatch'] * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
                    if_clamp_coeff=self.cfg.MODEL_LIGHT.if_clamp_coeff)
                return_dict.update({'envmapsPredScaledImage': envmapsPredScaledImage, 'envmapsPredScaledImage_offset_log_': envmapsPredScaledImage_offset_log_})
            elif self.cfg.MODEL_LIGHT.if_align_rerendering_envmap:
                pass
            else:
                envmapsPredScaledImage = models_brdf.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                    input_dict['envmapsBatch'] * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
                    if_clamp_coeff=self.cfg.MODEL_LIGHT.if_clamp_coeff)
                envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
                return_dict.update({'envmapsPredScaledImage': envmapsPredScaledImage, 'envmapsPredScaledImage_offset_log_': envmapsPredScaledImage_offset_log_})

        return_dict.update({'envmapsPredImage': envmapsPredImage, \
            'segEnvBatch': segEnvBatch, \
            'imBatchSmall': imBatchSmall, 'segBRDFBatchSmall': segBRDFBatchSmall, 'pixelNum_recon': pixelNum_recon}) 

        # Compute the rendered error
        pixelNum_render = max( (torch.sum(segBRDFBatchSmall ).cpu().data).item(), 1e-5 )
        
        normal_input, rough_input = normalInput, roughInput
        if self.cfg.DATA.if_pad_to_32x:
            normal_input = normal_input[:, :, :im_h, :im_w]
            rough_input = rough_input[:, :, :im_h, :im_w]
            albedoInput = albedoInput[:, :, :im_h, :im_w]

        envmapsImage_input = envmapsPredImage

        diffusePred, specularPred = self.non_learnable_layers['renderLayer'].forwardEnv(normalPred=normal_input.detach(), envmap=envmapsImage_input, diffusePred=albedoInput.detach(), roughPred=rough_input.detach())

        if self.cfg.MODEL_LIGHT.use_scale_aware_loss:
            diffusePredScaled, specularPredScaled = diffusePred, specularPred
        else:
            diffusePredScaled, specularPredScaled, _ = models_brdf.LSregressDiffSpec(
                diffusePred.detach(),
                specularPred.detach(),
                imBatchSmall,
                diffusePred, specularPred )

        renderedImPred_hdr = diffusePredScaled + specularPredScaled
        renderedImPred = renderedImPred_hdr
        renderedImPred_sdr = torch.clamp(renderedImPred_hdr ** (1.0/2.2), 0, 1)

        if self.cfg.MODEL_LIGHT.if_align_rerendering_envmap:
            cDiff, cSpec = (torch.sum(diffusePredScaled) / torch.sum(diffusePred)).data.item(), ((torch.sum(specularPredScaled) ) / (torch.sum(specularPred) ) ).data.item()
            # if cSpec == 0:
            cAlbedo = 1/ axisPred_ori.max().data.item()
            cLight = cDiff / cAlbedo
            # else:
            #     cLight = cSpec
            #     cAlbedo = cDiff / cLight
            #     cAlbedo = np.clip(cAlbedo, 1e-3, 1 / axisPred_ori.max().data.item() )
            #     cLight = cDiff / cAlbedo
            envmapsPredScaledImage = envmapsPredImage * cLight
            ic(cLight)
            envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
            return_dict.update({'envmapsPredScaledImage': envmapsPredScaledImage, 'envmapsPredScaledImage_offset_log_': envmapsPredScaledImage_offset_log_})

        return_dict.update({'renderedImPred': renderedImPred, 'renderedImPred_sdr': renderedImPred_sdr, 'pixelNum_render': pixelNum_render}) 
        return_dict.update({'LightNet_preds': {'axisPred': axisPred_ori, 'lambPred': lambPred_ori, 'weightPred': weightPred_ori, 'segEnvBatch': segEnvBatch, 'notDarkEnv': notDarkEnv}})

        return return_dict

    def forward_light_real(self, input_dict, return_dict_brdf={}):
        imBatch = input_dict['imBatch']
        assert imBatch.shape[0]==1

        im_h, im_w = input_dict['im_h_resized_to'], input_dict['im_w_resized_to']
        # im_h, im_w = self.cfg.DATA.im_height_padded_to, self.cfg.DATA.im_width_padded_to
        im_h = im_h//2*2
        im_w = im_w//2*2

        renderLayer = models_light.renderingLayer(isCuda = self.opt.if_cuda, 
            imWidth=im_w//2, imHeight=im_h//2,  
            envWidth = self.opt.cfg.MODEL_LIGHT.envWidth, envHeight = self.opt.cfg.MODEL_LIGHT.envHeight)
        output2env = models_light.output2env(isCuda = self.opt.if_cuda, 
            envWidth = self.opt.cfg.MODEL_LIGHT.envWidth, envHeight = self.opt.cfg.MODEL_LIGHT.envHeight, SGNum = self.opt.cfg.MODEL_LIGHT.SGNum )

        # Normalize Albedo and depth
        if not self.opt.cfg.DEBUG.if_load_dump_BRDF_offline:
            albedoInput = return_dict_brdf['albedoPred'].detach().clone()
            depthInput = return_dict_brdf['depthPred'].detach().clone()
            normalInput = return_dict_brdf['normalPred'].detach().clone()
            roughInput = return_dict_brdf['roughPred'].detach().clone()
        else:
            albedoInput = input_dict['albedoBatch'].detach().clone()
            depthInput = input_dict['depthBatch'].detach().clone()
            normalInput = input_dict['normalBatch'].detach().clone()
            roughInput = input_dict['roughBatch'].detach().clone()

        segBRDFBatch = input_dict['segBRDFBatch']
        pad_mask = input_dict['pad_mask'].float()

        albedoInput[:, :, :im_h, :im_w] = albedoInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(albedoInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0
        albedoInput = albedoInput * pad_mask
        # albedoInput[:, :, im_h:, :] = 0.
        # albedoInput[:, :, :, im_w:] = 0.
        # albedoInput[:, :, im_h:, im_w:] = 0.

        # print(torch.max(depthInput), torch.min(depthInput), torch.median(depthInput)) # [1, inf]
        # depthInput = torch.clamp(depthInput, 0., self.cfg.MODEL_LIGHT.depth_thres)
        # print(torch.mean(depthInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True))
        depthInput[:, :, :im_h, :im_w] = depthInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(depthInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0
        depthInput = depthInput * pad_mask
        # print(depthInput.shape, im_h, im_w)
        # depthInput[:, :, im_h:, :] = 0.
        # depthInput[:, :, :, im_w:] = 0.
        # depthInput[:, :, im_h:, im_w:] = 0.
        # print(torch.max(depthInput), torch.min(depthInput), torch.median(depthInput)) # [1, inf]

        normalInput[:, :, :im_h, :im_w] =  0.5 * (normalInput[:, :, :im_h, :im_w] + 1)
        normalInput = normalInput * pad_mask
        # normalInput[:, :, im_h:, :] = 0.
        # normalInput[:, :, :, im_w:] = 0.
        # normalInput[:, :, im_h:, im_w:] = 0.

        roughInput[:, :, :im_h, :im_w] =  0.5 * (roughInput[:, :, :im_h, :im_w] + 1)
        roughInput = roughInput * pad_mask
        # roughInput[:, :, im_h:, :] = 0.
        # roughInput[:, :, :, im_w:] = 0.
        # roughInput[:, :, im_h:, im_w:] = 0.
        
        x_stage1 = torch.cat([imBatch, albedoInput, normalInput, roughInput, depthInput ], dim=1 ).detach()

        # output_dict_model_all = self.MODEL_ALL.forward_light(x_stage1)
        output_dict_model_all = self.forward_LightNet_func(x_stage1)

        axisPred_ori, lambPred_ori, weightPred_ori = output_dict_model_all['axis'], output_dict_model_all['lamb'], output_dict_model_all['weight']

        if self.cfg.DATA.if_pad_to_32x:
            axisPred_ori, lambPred_ori, weightPred_ori = axisPred_ori[:, :, :, :im_h//2, :im_w//2], lambPred_ori[:, :, :im_h//2, :im_w//2], weightPred_ori[:, :, :im_h//2, :im_w//2]
            imBatch = imBatch[:, :, :im_h, :im_w]
            segBRDFBatch = segBRDFBatch[:, :, :im_h, :im_w]

        # imBatchSmall = F.adaptive_avg_pool2d(imBatch, (self.cfg.MODEL_LIGHT.envRow, self.cfg.MODEL_LIGHT.envCol) )
        imBatchSmall = F.interpolate(imBatch, scale_factor=0.5, mode="bilinear")
        # segBRDFBatchSmall = F.interpolate(segBRDFBatch, scale_factor=0.5, mode="nearest")
        # notDarkEnv = (torch.mean(torch.mean(torch.mean(input_dict['envmapsBatch'], 4), 4), 1, True ) > 0.001 ).float()
        # segEnvBatch = (segBRDFBatchSmall * input_dict['envmapsIndBatch'].expand_as(segBRDFBatchSmall) ).unsqueeze(-1).unsqueeze(-1)
        # segEnvBatch = segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)
        
        return_dict = {}

        envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred_ori, lambPred_ori, weightPred_ori, if_postprocessing=not self.cfg.MODEL_LIGHT.use_GT_light_sg) # all half red
        # print(axisPred_ori.shape, envmapsPredImage.shape)

        normal_input, rough_input = normalInput, roughInput
        if self.cfg.DATA.if_pad_to_32x:
            normal_input = normal_input[:, :, :im_h, :im_w]
            rough_input = rough_input[:, :, :im_h, :im_w]
            albedoInput = albedoInput[:, :, :im_h, :im_w]

        envmapsImage_input = envmapsPredImage

        # print(im_h, im_w, normal_input.shape, envmapsImage_input.shape, albedoInput.shape, rough_input.shape, )
        diffusePred, specularPred = renderLayer.forwardEnv(normalPred=normal_input.detach(), envmap=envmapsImage_input, diffusePred=albedoInput.detach(), roughPred=rough_input.detach())

        if self.cfg.MODEL_LIGHT.use_scale_aware_loss:
            assert False
            diffusePredScaled, specularPredScaled = diffusePred, specularPred
        else:
            diffusePredScaled, specularPredScaled, coefIm = models_brdf.LSregressDiffSpec(
                diffusePred.detach(),
                specularPred.detach(),
                imBatchSmall,
                diffusePred, specularPred )

        renderedImPred_hdr = diffusePredScaled + specularPredScaled
        renderedImPred = renderedImPred_hdr
        renderedImPred_sdr = torch.clamp(renderedImPred_hdr ** (1.0/2.2), 0, 1)

        cDiff, cSpec = (torch.sum(diffusePredScaled) / torch.sum(diffusePred)).data.item(), ((torch.sum(specularPredScaled) ) / (torch.sum(specularPred) ) ).data.item()
        if cSpec < 1e-3:
            cAlbedo = 1/ axisPred_ori.max().data.item()
            cLight = cDiff / cAlbedo
        else:
            cLight = cSpec
            cAlbedo = cDiff / cLight
            cAlbedo = np.clip(cAlbedo, 1e-3, 1 / axisPred_ori.max().data.item() )
            cLight = cDiff / cAlbedo
        envmapsPredImage = envmapsPredImage * cLight
        # envmapsPredImage = envmapsPredImage / 5.
        ic(torch.max(envmapsPredImage), torch.min(envmapsPredImage), torch.mean(envmapsPredImage), torch.median(envmapsPredImage))
        ic(cLight, cDiff, cSpec)

        return_dict.update({'imBatchSmall': imBatchSmall, 'envmapsPredImage': envmapsPredImage, 'renderedImPred': renderedImPred, 'renderedImPred_sdr': renderedImPred_sdr}) 
        return_dict.update({'LightNet_preds': {'axisPred': axisPred_ori, 'lambPred': lambPred_ori, 'weightPred': weightPred_ori}})
        return_dict.update({'coefIm': coefIm})

        return return_dict

    
    def forward_LightNet_UNet_func(self, x_stage1):
        x_stage1Large = F.interpolate(x_stage1, scale_factor=2, mode='bilinear')

        if self.opt.cascadeLevel == 0:
            # print(input_batch.shape)
            x1, x2, x3, x4, x5, x6 = self.LIGHT_Net['lightEncoder'](x_stage1Large)
        else:
            assert False
            # assert self.opt.cascadeLevel > 0
            # x1, x2, x3, x4, x5, x6 = self.LIGHT_Net['lightEncoder'](x_stage1, input_dict['envmapsPreBatch'].detach() )

        # print(input_batch.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape) # torch.Size([4, 11, 480, 640]) torch.Size([4, 128, 60, 80]) torch.Size([4, 256, 30, 40]) torch.Size([4, 256, 15, 20]) torch.Size([4, 512, 7, 10]) torch.Size([4, 512, 3, 5]) torch.Size([4, 1024, 3, 5])

        # Prediction
        # if 'axis' in self.cfg.MODEL_LIGHT.enable_list and not self.cfg.MODEL_LIGHT.use_GT_light_sg:
        axisPred_ori = self.LIGHT_Net['axisDecoder'](x1, x2, x3, x4, x5, x6) # torch.Size([4, 12, 3, 120, 160])
        lambPred_ori = self.LIGHT_Net['lambDecoder'](x1, x2, x3, x4, x5, x6) # torch.Size([4, 12, 120, 160])
        weightPred_ori = self.LIGHT_Net['weightDecoder'](x1, x2, x3, x4, x5, x6) # torch.Size([4, 36, 120, 160])
        return {'axis': axisPred_ori, 'lamb': lambPred_ori, 'weight': weightPred_ori}

    # def freeze_BN(self):
    #     freeze_bn_in_module(self.MODEL_ALL._['ro'])

    def freeze_BRDF(self):
        self.turn_off_names(['shared_encoder_stage0'])
        freeze_bn_in_module(self.MODEL_ALL._.shared_encoder_stage0)
        if self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities:
            self.turn_off_names(['shared_BRDF_decoder'])
            freeze_bn_in_module(self.MODEL_ALL._.shared_BRDF_decoder)
        if self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_pretrained_over_BRDF_modalities:
            self.turn_off_names(['shared_BRDF_pretrained'])
            freeze_bn_in_module(self.MODEL_ALL._.shared_BRDF_pretrained)
        if 'al' in self.modalities_stage0:
            self.turn_off_names(['_.al'])
            freeze_bn_in_module(self.MODEL_ALL._['al'])
        if 'no' in self.modalities_stage0:
            self.turn_off_names(['_.no'])
            freeze_bn_in_module(self.MODEL_ALL._['no'])
        if 'de' in self.modalities_stage0:
            self.turn_off_names(['_.de'])
            freeze_bn_in_module(self.MODEL_ALL._['de'])
        if 'ro' in self.modalities_stage0:
            self.turn_off_names(['_.ro'])
            freeze_bn_in_module(self.MODEL_ALL._['ro'])
    
    def freeze_BRDF_except_albedo(self, if_print=True):
        if 'no' in self.modalities_stage0:
            self.turn_off_names(['_.no'], if_print=if_print)
            freeze_bn_in_module(self.MODEL_ALL._['no'], if_print=if_print)
        if 'de' in self.modalities_stage0:
            self.turn_off_names(['_.de'], if_print=if_print)
            freeze_bn_in_module(self.MODEL_ALL._['de'], if_print=if_print)
        if 'ro' in self.modalities_stage0:
            self.turn_off_names(['_.ro'], if_print=if_print)
            freeze_bn_in_module(self.MODEL_ALL._['ro'], if_print=if_print)

    def unfreeze_BRDF_except_albedo(self, if_print=True):
        if 'no' in self.modalities_stage0:
            self.turn_on_names(['_.no'], if_print=if_print)
            unfreeze_bn_in_module(self.MODEL_ALL._['no'], if_print=if_print)
        if 'de' in self.modalities_stage0:
            self.turn_on_names(['_.de'], if_print=if_print)
            unfreeze_bn_in_module(self.MODEL_ALL._['de'], if_print=if_print)
        if 'ro' in self.modalities_stage0:
            self.turn_on_names(['_.ro'], if_print=if_print)
            unfreeze_bn_in_module(self.MODEL_ALL._['ro'], if_print=if_print)

    def freeze_BRDF_except_depth_normal(self, if_print=True):
        if 'al' in self.modalities_stage0:
            self.turn_off_names(['_.al'], if_print=if_print)
            freeze_bn_in_module(self.MODEL_ALL._['al'], if_print=if_print)
        if 'ro' in self.modalities_stage0:
            self.turn_off_names(['_.ro'], if_print=if_print)
            freeze_bn_in_module(self.MODEL_ALL._['ro'], if_print=if_print)

    def unfreeze_BRDF_except_depth_normal(self, if_print=True):
        if 'al' in self.modalities_stage0:
            self.turn_on_names(['_.al'], if_print=if_print)
            unfreeze_bn_in_module(self.MODEL_ALL._['al'], if_print=if_print)
        if 'ro' in self.modalities_stage0:
            self.turn_on_names(['_.ro'], if_print=if_print)
            unfreeze_bn_in_module(self.MODEL_ALL._['ro'], if_print=if_print)

    def print_net(self):
        count_grads = 0
        for name, param in self.named_parameters():
            if_trainable_str = white_blue('True') if param.requires_grad else green('False')
            self.logger.info(name + str(param.shape) + ' ' + if_trainable_str)
            if param.requires_grad:
                count_grads += 1
        self.logger.info(magenta('---> ALL %d params; %d trainable'%(len(list(self.named_parameters())), count_grads)))
        return count_grads

    def turn_off_all_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.logger.info(colored('only_enable_camH_bboxPredictor', 'white', 'on_red'))

    def turn_on_all_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
        self.logger.info(colored('turned on all params', 'white', 'on_red'))

    def turn_on_names(self, in_names, if_print=True):
        for name, param in self.named_parameters():
            for in_name in in_names:
            # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if in_name in name:
                    param.requires_grad = True
                    if if_print:
                        self.logger.info(colored('turn_ON_names: ' + in_name, 'white', 'on_red'))

    def turn_off_names(self, in_names, exclude_names=[], if_print=True):
        for name, param in self.named_parameters():
            for in_name in in_names:
            # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if_not_in_exclude = all([exclude_name not in name for exclude_name in exclude_names]) # any item in exclude_names must not be in the paramater name
                if in_name in name and if_not_in_exclude:
                    param.requires_grad = False
                    if if_print:
                        self.logger.info(colored('turn_OFF_names: ' + in_name, 'white', 'on_red'))

    def freeze_bn_semantics(self):
        freeze_bn_in_module(self.SEMSEG_Net)

    def freeze_bn_matseg(self):
        freeze_bn_in_module(self.MATSEG_Net)

    def load_pretrained_MODEL_BRDF(self, if_load_encoder=True, if_load_decoder=True, if_load_Bs=True):
        # if self.opt.if_cluster:
        #     pretrained_path_root = Path('/viscompfs/users/ruizhu/models_ckpt/')
        # else:
        #     pretrained_path_root = Path('/home/ruizhu/Documents/Projects/semanticInverse/models_ckpt/')
        pretrained_path_root = Path(self.opt.cfg.PATH.models_ckpt_path)
        # loaded_strings = []
        module_names = []
        if if_load_encoder:
            module_names.append('encoder')    
        if if_load_decoder:
            module_names += ['albedoDecoder', 'normalDecoder', 'roughDecoder', 'depthDecoder']
        if if_load_Bs:
            # module_names += ['albedoBs', 'normalBs', 'roughBs', 'depthBs']
            if 'al' in self.modalities_stage0:
                module_names += ['albedoBs']
            if 'no' in self.modalities_stage0:
                module_names += ['normalBs']
            if 'ro' in self.modalities_stage0:
                module_names += ['roughBs']
            if 'de' in self.modalities_stage0:
                module_names += ['depthBs']

        saved_names_dict = {
            'encoder': 'encoder', 
            'albedoDecoder': 'albedo', 
            'normalDecoder': 'normal', 
            'roughDecoder': 'rough', 
            'depthDecoder': 'depth', 
            'albedoBs': 'albedoBs', 
            'normalBs': 'normalBs', 
            'roughBs': 'roughBs', 
            'depthBs': 'depthBs'
        }
        pretrained_pth_name_dict = {
            'encoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'albedoDecoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'normalDecoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'roughDecoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'depthDecoder': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_BRDF_cascade0, 
            'albedoBs': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_Bs_cascade0, 
            'normalBs': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_Bs_cascade0, 
            'roughBs': self.opt.cfg.MODEL_BRDF.pretrained_pth_name_Bs_cascade0, 
            'depthBs':self.opt.cfg.MODEL_BRDF.pretrained_pth_name_Bs_cascade0
        }
        for module_name in module_names:
            saved_name = saved_names_dict[module_name]
            pickle_path = str(pretrained_path_root / pretrained_pth_name_dict[module_name]) % saved_name
            print('Loading ' + pickle_path)
            self.BRDF_Net[module_name].load_state_dict(
                torch.load(pickle_path).state_dict())
            # loaded_strings.append(saved_name)

            self.logger.info(magenta('Loaded pretrained BRDFNet-%s from %s'%(module_name, pickle_path)))
