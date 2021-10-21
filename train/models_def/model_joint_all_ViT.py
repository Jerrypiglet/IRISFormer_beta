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

from models_def.model_dpt.models_multi_head_ViT_DPT import ModelAll_ViT

class Model_Joint_ViT(nn.Module):
    def __init__(self, opt, logger):
        super(Model_Joint_ViT, self).__init__()
        self.opt = opt
        self.cfg = opt.cfg
        self.logger = logger

        assert self.opt.cfg.MODEL_ALL.ViT_baseline.enable

        self.MODEL_ALL = ModelAll_ViT(
            opt=opt, 
            modalities=self.cfg.MODEL_ALL.enable_list, 
            backbone="vitb_rn50_384_N_layers", 
            N_layers_encoder = self.cfg.MODEL_ALL.ViT_baseline.N_layers_encoder, 
            N_layers_decoder = self.cfg.MODEL_ALL.ViT_baseline.N_layers_decoder
        )

    def forward(self, input_dict):
        assert self.opt.cfg.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities
        # module_hooks_dict = {}
        # input_dict_extra = {}
        # output_dict = {}

        # print(input_dict['input_batch_brdf'].shape)
        # input_dict_extra['shared_encoder_outputs'] = forward_vit_ViT(
        #     self.opt, self.opt.cfg.MODEL_ALL.ViT_baseline, self.MODEL_ALL.shared_encoder, 
        #     input_dict['input_batch_brdf'])

        output_dict_model_all = self.MODEL_ALL(input_dict['input_batch_brdf'])

        return_dict = {}

        modalities = self.opt.cfg.MODEL_ALL.enable_list
        for modality in modalities:
            # vit_out = self.MODEL_ALL[modality].forward(None, input_dict_extra=input_dict_extra)
            vit_out = output_dict_model_all[modality]

            if modality == 'al':
                albedoPred = 0.5 * (vit_out + 1)
                input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
                albedoPred = torch.clamp(albedoPred, 0, 1)
                albedoPred_aligned = models_brdf.LSregress(albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred),
                        input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoPred)
                albedoPred_aligned = torch.clamp(albedoPred_aligned, 0, 1)
                return_dict.update({'albedoPred': albedoPred, 'albedoPred_aligned': albedoPred_aligned})
            elif modality == 'de':
                depthPred = vit_out
                depthPred_aligned = models_brdf.LSregress(depthPred *  input_dict['segAllBatch'].expand_as(depthPred),
                        input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthPred)
                return_dict.update({'depthPred': depthPred, 'depthPred_aligned': depthPred_aligned})
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