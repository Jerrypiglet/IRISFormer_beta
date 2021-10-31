from math import e
import torch
import torch.nn as nn

from models_def.model_matseg import Baseline
from utils.utils_misc import *
# import pac
from utils.utils_training import freeze_bn_in_module
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
        self.modalities_stage1 = list(set(self.modalities) & set(['axis', 'lamb', 'weight']))
        self.if_BRDF = self.modalities_stage0 != []
        self.if_Light = self.modalities_stage1 != []
        self.load_brdf_gt = self.opt.cfg.DATA.load_brdf_gt
        self.use_GT_brdf = self.opt.cfg.MODEL_LIGHT.use_GT_brdf

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

        if self.if_Light:
            self.non_learnable_layers = {}
            self.non_learnable_layers['renderLayer'] = models_light.renderingLayer(isCuda = opt.if_cuda, 
                imWidth=opt.cfg.MODEL_LIGHT.envCol, imHeight=opt.cfg.MODEL_LIGHT.envRow, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight)
            self.non_learnable_layers['output2env'] = models_light.output2env(isCuda = opt.if_cuda, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight, SGNum = opt.cfg.MODEL_LIGHT.SGNum )


            if self.cfg.MODEL_LIGHT.freeze_BRDF_Net and self.if_BRDF:
                self.freeze_BRDF()

    def forward(self, input_dict):
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
            return_dict_brdf = self.forward_brdf(x_stage0, input_dict)
            output_dict.update(return_dict_brdf)
        else:
            return_dict_brdf = {}

        if self.if_Light:
            if self.opt.cfg.DATASET.if_no_gt_BRDF:
                return_dict_light = self.forward_light_real(input_dict, return_dict_brdf)
            else:
                return_dict_light = self.forward_light(input_dict, return_dict_brdf)
            output_dict.update(return_dict_light)

        return output_dict


    def forward_brdf(self, x_stage0, input_dict):
        output_dict_model_all = self.MODEL_ALL.forward_brdf(x_stage0)

        return_dict = {}

        for modality in self.modalities_stage0:
            # vit_out = self.MODEL_ALL[modality].forward(None, input_dict_extra=input_dict_extra)
            vit_out = output_dict_model_all[modality]

            if modality == 'al':
                albedoPred = 0.5 * (vit_out + 1)
                albedoPred = torch.clamp(albedoPred, 0, 1)
                return_dict.update({'albedoPred': albedoPred})
                if (not self.opt.cfg.DATASET.if_no_gt_BRDF) and self.load_brdf_gt:
                    input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
                    albedoPred_aligned = models_brdf.LSregress(albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred),
                            input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoPred)
                    albedoPred_aligned = torch.clamp(albedoPred_aligned, 0, 1)
                    return_dict.update({'albedoPred_aligned': albedoPred_aligned})
            elif modality == 'de':
                if self.opt.cfg.MODEL_BRDF.loss.depth.if_use_Zhengqin_loss:
                    depthPred = vit_out
                elif self.opt.cfg.MODEL_BRDF.loss.depth.if_use_midas_loss:
                    return_dict.update({'depthInvPred': vit_out})
                    depthPred = 1. / (vit_out + 1e-8)
                return_dict.update({'depthPred': depthPred})
                if (not self.opt.cfg.DATASET.if_no_gt_BRDF) and self.load_brdf_gt:
                    depthPred_aligned = models_brdf.LSregress(depthPred *  input_dict['segAllBatch'].expand_as(depthPred),
                            input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthPred)
                    return_dict.update({'depthPred_aligned': depthPred_aligned})
            elif modality == 'ro':
                roughPred = vit_out
                return_dict.update({'roughPred': roughPred})
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

        # Normalize Albedo and depth
        if 'al' in self.modalities_stage0 and not self.use_GT_brdf:
            albedoInput = return_dict_brdf['albedoPred'].detach().clone()
        else:
            albedoInput = input_dict['albedoBatch'].detach().clone()

        if 'de' in self.modalities_stage0 and not self.use_GT_brdf:
            depthInput = return_dict_brdf['depthPred'].detach().clone()
        else:
            depthInput = input_dict['depthBatch'].detach().clone()

        if 'no' in self.modalities_stage0 and not self.use_GT_brdf:
            normalInput = return_dict_brdf['normalPred'].detach().clone()
        else:
            normalInput = input_dict['normalBatch'].detach().clone()

        if 'ro' in self.modalities_stage0 and not self.use_GT_brdf:
            roughInput = return_dict_brdf['roughPred'].detach().clone()
        else:
            roughInput = input_dict['roughBatch'].detach().clone()

        imBatch = input_dict['imBatch']
        segBRDFBatch = input_dict['segBRDFBatch']


        albedoInput[:, :, :im_h, :im_w] = albedoInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(albedoInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0

        depthInput[:, :, :im_h, :im_w] = depthInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(depthInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0
        normalInput[:, :, :im_h, :im_w] =  0.5 * (normalInput[:, :, :im_h, :im_w] + 1)
        roughInput[:, :, :im_h, :im_w] =  0.5 * (roughInput[:, :, :im_h, :im_w] + 1)
        
        x_stage1 = torch.cat([imBatch, albedoInput, normalInput, roughInput, depthInput ], dim=1 ).detach()

        output_dict_model_all = self.MODEL_ALL.forward_light(x_stage1)

        axisPred_ori, lambPred_ori, weightPred_ori = output_dict_model_all['axis'], output_dict_model_all['lamb'], output_dict_model_all['weight']

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
            envmapsPredScaledImage_log = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        elif self.cfg.MODEL_LIGHT.use_GT_light_envmap:
            envmapsPredScaledImage = envmapsPredImage # gt envmap already scaled in dataloader
            envmapsPredScaledImage_log = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        elif self.cfg.MODEL_LIGHT.use_scale_aware_loss:
            envmapsPredScaledImage = envmapsPredImage # not aligning envmap
            envmapsPredScaledImage_log = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        else: # scale-invariant
            if self.cfg.MODEL_LIGHT.if_align_log_envmap:
                envmapsPredScaledImage = models_brdf.LSregress(torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset).detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                    torch.log(input_dict['envmapsBatch'] + self.cfg.MODEL_LIGHT.offset) * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
                    if_clamp_coeff=False)
                envmapsPredScaledImage_log = models_brdf.LSregress(torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset).detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                    torch.log(input_dict['envmapsBatch'] + self.cfg.MODEL_LIGHT.offset) * segEnvBatch.expand_as(input_dict['envmapsBatch']), torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset), 
                    if_clamp_coeff=False)
            else:
                envmapsPredScaledImage = models_brdf.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                    input_dict['envmapsBatch'] * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
                    if_clamp_coeff=False)
                envmapsPredScaledImage_log = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)

        return_dict.update({'envmapsPredImage': envmapsPredImage, 'envmapsPredScaledImage': envmapsPredScaledImage, 'envmapsPredScaledImage_log': envmapsPredScaledImage_log, \
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
            diffusePredScaled, specularPredScaled = models_brdf.LSregressDiffSpec(
                diffusePred.detach(),
                specularPred.detach(),
                imBatchSmall,
                diffusePred, specularPred )

        renderedImPred_hdr = diffusePredScaled + specularPredScaled
        renderedImPred = renderedImPred_hdr
        renderedImPred_sdr = torch.clamp(renderedImPred_hdr ** (1.0/2.2), 0, 1)

        return_dict.update({'renderedImPred': renderedImPred, 'renderedImPred_sdr': renderedImPred_sdr, 'pixelNum_render': pixelNum_render}) 
        return_dict.update({'LightNet_preds': {'axisPred': axisPred_ori, 'lambPred': lambPred_ori, 'weightPred': weightPred_ori, 'segEnvBatch': segEnvBatch, 'notDarkEnv': notDarkEnv}})

        return return_dict

    def forward_light_real(self, input_dict, return_dict_brdf={}):
        imBatch = input_dict['imBatch']
        assert imBatch.shape[0]==1
        im_h, im_w = input_dict['im_h_resized_to'], input_dict['im_w_resized_to']
        # Normalize Albedo and depth
        albedoInput = return_dict_brdf['albedoPred'].detach().clone()
        depthInput = return_dict_brdf['depthPred'].detach().clone()
        normalInput = return_dict_brdf['normalPred'].detach().clone()
        roughInput = return_dict_brdf['roughPred'].detach().clone()

        # segBRDFBatch = input_dict['segBRDFBatch']
        pad_mask = input_dict['pad_mask'].float()

        # albedoInput_masked = albedoInput * pad_mask.unsqueeze(1)
        # albedoInput_unmasked = albedoInput * (1. - pad_mask.unsqueeze(1))
        # albedoInput_masked_mean = torch.sum(albedoInput_masked.flatten(1), dim=1, keepdim=True) / (torch.sum(pad_mask.flatten(1), dim=1, keepdim=True) * 3.)
        # albedoInput = albedoInput_masked / torch.clamp(albedoInput_masked_mean, min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0 + albedoInput_unmasked
        albedoInput[:, :, :im_h, :im_w] = albedoInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(albedoInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0

        # depthInput_masked = depthInput * pad_mask.unsqueeze(1)
        # depthInput_unmasked = depthInput * (1. - pad_mask.unsqueeze(1))
        # depthInput_masked_mean = torch.sum(depthInput_masked.flatten(1), dim=1, keepdim=True) / (torch.sum(pad_mask.flatten(1), dim=1, keepdim=True) * 3.)
        # depthInput = depthInput_masked / torch.clamp(depthInput_masked_mean, min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0 + depthInput_unmasked
        depthInput[:, :, :im_h, :im_w] = depthInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(depthInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0

        normalInput[:, :, :im_h, :im_w] =  0.5 * (normalInput[:, :, :im_h, :im_w] + 1)
        # normalInput_masked = normalInput * pad_mask.unsqueeze(1)
        # normalInput_unmasked = normalInput * (1. - pad_mask.unsqueeze(1))
        # normalInput = 0.5 * (normalInput_masked + 1.) + normalInput_unmasked

        roughInput[:, :, :im_h, :im_w] =  0.5 * (roughInput[:, :, :im_h, :im_w] + 1)
        # roughInput_masked = roughInput * pad_mask.unsqueeze(1)
        # roughInput_unmasked = roughInput * (1. - pad_mask.unsqueeze(1))
        # roughInput = 0.5 * (roughInput_masked + 1.) + roughInput_unmasked
        
        x_stage1 = torch.cat([imBatch, albedoInput, normalInput, roughInput, depthInput ], dim=1 ).detach()

        output_dict_model_all = self.MODEL_ALL.forward_light(x_stage1)

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

        envmapsPredImage, axisPred, lambPred, weightPred = self.non_learnable_layers['output2env'].output2env(axisPred_ori, lambPred_ori, weightPred_ori, if_postprocessing=not self.cfg.MODEL_LIGHT.use_GT_light_sg)

        # pixelNum_recon = max( (torch.sum(segEnvBatch ).cpu().data).item(), 1e-5)
        # if self.cfg.MODEL_LIGHT.use_GT_light_sg:
        #     envmapsPredScaledImage = envmapsPredImage * (input_dict['hdr_scaleBatch'].flatten().view(-1, 1, 1, 1, 1, 1))
        #     envmapsPredScaledImage_log = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        # elif self.cfg.MODEL_LIGHT.use_GT_light_envmap:
        #     envmapsPredScaledImage = envmapsPredImage # gt envmap already scaled in dataloader
        #     envmapsPredScaledImage_log = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        # elif self.cfg.MODEL_LIGHT.use_scale_aware_loss:
        #     envmapsPredScaledImage = envmapsPredImage # not aligning envmap
        #     envmapsPredScaledImage_log = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)
        # else: # scale-invariant
        #     if self.cfg.MODEL_LIGHT.if_align_log_envmap:
        #         envmapsPredScaledImage = models_brdf.LSregress(torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset).detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
        #             torch.log(input_dict['envmapsBatch'] + self.cfg.MODEL_LIGHT.offset) * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
        #             if_clamp_coeff=False)
        #         envmapsPredScaledImage_log = models_brdf.LSregress(torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset).detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
        #             torch.log(input_dict['envmapsBatch'] + self.cfg.MODEL_LIGHT.offset) * segEnvBatch.expand_as(input_dict['envmapsBatch']), torch.log(envmapsPredImage + self.cfg.MODEL_LIGHT.offset), 
        #             if_clamp_coeff=False)
        #     else:
        #         envmapsPredScaledImage = models_brdf.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
        #             input_dict['envmapsBatch'] * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
        #             if_clamp_coeff=False)
        #         envmapsPredScaledImage_log = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)

        # return_dict.update({'envmapsPredImage': envmapsPredImage, 'envmapsPredScaledImage': envmapsPredScaledImage, 'envmapsPredScaledImage_log': envmapsPredScaledImage_log, \
        #     'segEnvBatch': segEnvBatch, \
        #     'imBatchSmall': imBatchSmall, 'segBRDFBatchSmall': segBRDFBatchSmall, 'pixelNum_recon': pixelNum_recon}) 

        # # Compute the rendered error
        # pixelNum_render = max( (torch.sum(segBRDFBatchSmall ).cpu().data).item(), 1e-5 )
        
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
            diffusePredScaled, specularPredScaled = models_brdf.LSregressDiffSpec(
                diffusePred.detach(),
                specularPred.detach(),
                imBatchSmall,
                diffusePred, specularPred )

        renderedImPred_hdr = diffusePredScaled + specularPredScaled
        renderedImPred = renderedImPred_hdr
        renderedImPred_sdr = torch.clamp(renderedImPred_hdr ** (1.0/2.2), 0, 1)

        return_dict.update({'renderedImPred': renderedImPred, 'renderedImPred_sdr': renderedImPred_sdr}) 
        return_dict.update({'LightNet_preds': {'axisPred': axisPred_ori, 'lambPred': lambPred_ori, 'weightPred': weightPred_ori}})

        return return_dict

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

    def turn_on_names(self, in_names):
        for name, param in self.named_parameters():
            for in_name in in_names:
            # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if in_name in name:
                    param.requires_grad = True
                    self.logger.info(colored('turn_ON_names: ' + in_name, 'white', 'on_red'))

    def turn_off_names(self, in_names, exclude_names=[]):
        for name, param in self.named_parameters():
            for in_name in in_names:
            # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if_not_in_exclude = all([exclude_name not in name for exclude_name in exclude_names]) # any item in exclude_names must not be in the paramater name
                if in_name in name and if_not_in_exclude:
                    param.requires_grad = False
                    self.logger.info(colored('turn_OFF_names: ' + in_name, 'white', 'on_red'))

    def freeze_bn_semantics(self):
        freeze_bn_in_module(self.SEMSEG_Net)

    def freeze_bn_matseg(self):
        freeze_bn_in_module(self.MATSEG_Net)