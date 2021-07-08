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
import models_def.models_brdf_pac_pool as models_brdf_pac_pool
import models_def.models_brdf_pac_conv as models_brdf_pac_conv
import models_def.models_brdf_safenet as models_brdf_safenet
import models_def.models_light as models_light 
import models_def.models_layout_emitter as models_layout_emitter
import models_def.models_object_detection as models_object_detection
import models_def.models_mesh_reconstruction as models_mesh_reconstruction
import models_def.models_layout_emitter_lightAccu as models_layout_emitter_lightAccu
import models_def.models_layout_emitter_lightAccuScatter as models_layout_emitter_lightAccuScatter
import models_def.model_matcls as model_matcls
import models_def.model_nvidia.AppGMM as AppGMM

from SimpleLayout.SimpleSceneTorchBatch import SimpleSceneTorchBatch
from utils.utils_total3D.utils_OR_layout import get_layout_bdb_sunrgbd
from utils.utils_total3D.utils_OR_cam import get_rotation_matix_result

from icecream import ic

class Model_Joint(nn.Module):
    def __init__(self, opt, logger):
        super(Model_Joint, self).__init__()
        self.opt = opt
        self.cfg = opt.cfg
        self.logger = logger
        self.non_learnable_layers = {}

        if self.cfg.MODEL_MATSEG.enable:
            input_dim = 3 if not self.cfg.MODEL_MATSEG.use_semseg_as_input else 4
            self.MATSEG_Net = Baseline(self.cfg.MODEL_MATSEG, embed_dims=self.cfg.MODEL_MATSEG.embed_dims, input_dim=input_dim)

            if self.opt.cfg.MODEL_MATSEG.load_pretrained_pth:
                self.load_pretrained_matseg()

        if self.cfg.MODEL_SEMSEG.enable and (not self.cfg.MODEL_BRDF.enable_semseg_decoder):
            # value_scale = 255
            # mean = [0.485, 0.456, 0.406]
            # mean = [item * value_scale for item in mean]
            # self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(opt.device)
            # std = [0.229, 0.224, 0.225]
            # std = [item * value_scale for item in std]
            # self.std = torch.tensor(std).view(1, 3, 1, 1).to(opt.device)
            self.semseg_configs = self.opt.semseg_configs
            self.semseg_path = self.opt.cfg.MODEL_SEMSEG.semseg_path_cluster if opt.if_cluster else self.opt.cfg.MODEL_SEMSEG.semseg_path_local
            # self.MATSEG_Net = Baseline(self.cfg.MODEL_SEMSEG)
            assert self.semseg_configs.arch == 'psp'

            from models_def.models_semseg.pspnet import PSPNet
            self.SEMSEG_Net = PSPNet(layers=self.semseg_configs.layers, classes=self.semseg_configs.classes, zoom_factor=self.semseg_configs.zoom_factor, criterion=opt.semseg_criterion, pretrained=False)
            if opt.cfg.MODEL_SEMSEG.if_freeze:
                self.SEMSEG_Net.eval()

            if self.opt.cfg.MODEL_SEMSEG.load_pretrained_pth:
                self.load_pretrained_semseg()

        if self.cfg.MODEL_GMM.enable:
            self.MODEL_GMM = AppGMM.AppGMM(self.opt,)
            self.MODEL_GMM.set_optical_flow_model(self.opt)
            
        if self.cfg.MODEL_BRDF.enable:
            in_channels = 3
            if self.opt.cfg.MODEL_MATSEG.use_as_input:
                in_channels += 1
            if self.opt.cfg.MODEL_SEMSEG.use_as_input:
                in_channels += 1
            if self.opt.cfg.MODEL_MATSEG.use_pred_as_input:
                in_channels += 1

            self.decoder_to_use = models_brdf.decoder0

            self.if_semseg_matseg_guidance = self.opt.cfg.MODEL_MATSEG.if_guide or self.opt.cfg.MODEL_SEMSEG.if_guide
            if self.if_semseg_matseg_guidance:
                self.decoder_to_use = models_brdf.decoder0_guide

            if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_pool:
                self.decoder_to_use = models_brdf_pac_pool.decoder0_pacpool
            if self.opt.cfg.MODEL_MATSEG.if_albedo_pac_conv:
                self.decoder_to_use = models_brdf_pac_conv.decoder0_pacconv
            if self.opt.cfg.MODEL_MATSEG.if_albedo_safenet:
                self.decoder_to_use = models_brdf_safenet.decoder0_safenet

            self.BRDF_Net = nn.ModuleDict({
                    'encoder': models_brdf.encoder0(opt, cascadeLevel = self.opt.cascadeLevel, in_channels = in_channels)
                    })
            if self.cfg.MODEL_BRDF.enable_BRDF_decoders:
                if 'al' in self.cfg.MODEL_BRDF.enable_list:
                    self.BRDF_Net.update({'albedoDecoder': self.decoder_to_use(opt, mode=0)})
                if 'no' in self.cfg.MODEL_BRDF.enable_list:
                    self.BRDF_Net.update({'normalDecoder': self.decoder_to_use(opt, mode=1)})
                if 'ro' in self.cfg.MODEL_BRDF.enable_list:
                    self.BRDF_Net.update({'roughDecoder': self.decoder_to_use(opt, mode=2)})
                if 'de' in self.cfg.MODEL_BRDF.enable_list:
                    if self.cfg.MODEL_BRDF.use_scale_aware_depth:
                        assert self.cfg.MODEL_BRDF.depth_activation in ['relu', 'sigmoid']
                        if self.cfg.MODEL_BRDF.depth_activation == 'relu':
                            self.BRDF_Net.update({'depthDecoder': self.decoder_to_use(opt, mode=5)})
                        elif self.cfg.MODEL_BRDF.depth_activation == 'sigmoid':
                            self.BRDF_Net.update({'depthDecoder': self.decoder_to_use(opt, mode=6)})
                    else:
                        self.BRDF_Net.update({'depthDecoder': self.decoder_to_use(opt, mode=4)})
                    
            if self.cfg.MODEL_BRDF.enable_semseg_decoder:
                self.BRDF_Net.update({'semsegDecoder': self.decoder_to_use(opt, mode=-1, out_channel=self.cfg.MODEL_SEMSEG.semseg_classes, if_PPM=self.cfg.MODEL_BRDF.semseg_PPM)})

            if self.cfg.MODEL_BRDF.if_freeze:
                self.BRDF_Net.eval()

        # self.guide_net = guideNet(opt)
        if self.cfg.MODEL_LIGHT.load_pretrained_MODEL_BRDF:
            self.load_pretrained_MODEL_BRDF(self.cfg.MODEL_BRDF.pretrained_pth_name)

        if self.cfg.MODEL_LIGHT.enable:
            self.LIGHT_Net = nn.ModuleDict({})
            self.LIGHT_Net.update({'lightEncoder':  models_light.encoderLight(cascadeLevel = opt.cascadeLevel, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
            self.LIGHT_Net.update({'axisDecoder':  models_light.decoderLight(mode=0, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
            self.LIGHT_Net.update({'lambDecoder':  models_light.decoderLight(mode = 1, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
            self.LIGHT_Net.update({'weightDecoder':  models_light.decoderLight(mode = 2, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
            self.non_learnable_layers['renderLayer'] = models_light.renderingLayer(isCuda = opt.if_cuda, 
                imWidth=opt.cfg.MODEL_LIGHT.envCol, imHeight=opt.cfg.MODEL_LIGHT.envRow, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight)
            self.non_learnable_layers['output2env'] = models_light.output2env(isCuda = opt.if_cuda, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight, SGNum = opt.cfg.MODEL_LIGHT.SGNum )

            if not self.opt.cfg.MODEL_LIGHT.use_GT_brdf:

                if self.cfg.MODEL_LIGHT.freeze_BRDF_Net:
                    self.turn_off_names(['BRDF_Net'])
                    freeze_bn_in_module(self.BRDF_Net)

                if self.cfg.MODEL_LIGHT.if_freeze:
                    self.turn_off_names(['LIGHT_Net'])
                    freeze_bn_in_module(self.LIGHT_Net)

            if self.cfg.MODEL_LIGHT.load_pretrained_MODEL_LIGHT:
                self.load_pretrained_MODEL_LIGHT(self.cfg.MODEL_LIGHT.pretrained_pth_name)
        
        if self.cfg.MODEL_LAYOUT_EMITTER.enable:

            # the vanilla emitter/layout model: full FC, adapted from Total3D
            if_vanilla_emitter = 'em' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list and not(self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.enable)
            if_layout = 'lo' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list

            if if_layout or if_vanilla_emitter:
                if self.cfg.MODEL_LAYOUT_EMITTER.layout.if_indept_encoder:
                    self.LAYOUT_EMITTER_NET_encoder = models_brdf.encoder0(opt, cascadeLevel = 0, in_channels = 3, encoder_exclude = ['x5', 'x6'])
                    if self.cfg.MODEL_LAYOUT_EMITTER.layout.if_freeze:
                        self.turn_off_names(['LAYOUT_EMITTER_NET_encoder'])
                        freeze_bn_in_module(self.LAYOUT_EMITTER_NET_encoder)

                self.LAYOUT_EMITTER_NET_fc = models_layout_emitter.decoder_layout_emitter(opt, if_layout=if_layout, if_emitter_vanilla_fc=if_vanilla_emitter)
                if self.cfg.MODEL_LAYOUT_EMITTER.layout.if_freeze:
                    self.turn_off_names(['LAYOUT_EMITTER_NET_fc'])
                    freeze_bn_in_module(self.LAYOUT_EMITTER_NET_fc)
                    if self.cfg.MODEL_LAYOUT_EMITTER.layout.if_freeze_cls_heads:
                        self.turn_off_names(['LAYOUT_EMITTER_NET_fc'])
                        freeze_bn_in_module(self.LAYOUT_EMITTER_NET_fc)
                        

            if 'em' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.enable:
                    # --- lightAccuNet
                    self.EMITTER_LIGHT_ACCU_NET = models_layout_emitter_lightAccu.emitter_lightAccu(opt, envHeight=self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.envHeight, envWidth=self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.envWidth)
                    if self.cfg.MODEL_LAYOUT_EMITTER.emitter.if_freeze:
                        self.turn_off_names(['EMITTER_LIGHT_ACCU_NET'])
                        freeze_bn_in_module(self.EMITTER_LIGHT_ACCU_NET)

                    # --- emitterNet
                    if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.version == 'V1':
                        self.EMITTER_NET = models_layout_emitter_lightAccu.decoder_layout_emitter_lightAccu_(opt)
                    elif self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.version == 'V2':
                        input_channels = 3
                        if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_img_feats_as_input:                        
                            if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.sample_BRDF_feats_instead_of_learn_feats:
                                input_channels += self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.img_feats_channels # concat of BRDF feats
                            else:
                                input_channels += self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.img_feats_channels + 3 # +3 for img input
                        self.EMITTER_NET = models_layout_emitter_lightAccu.decoder_layout_emitter_lightAccu_UNet_V2(opt, input_channels=input_channels)
                    elif self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.version == 'V3':
                        self.EMITTER_NET = models_layout_emitter_lightAccuScatter.decoder_layout_emitter_lightAccuScatter_UNet_V3(opt)

                    if self.cfg.MODEL_LAYOUT_EMITTER.emitter.if_freeze:
                        self.turn_off_names(['EMITTER_NET'], exclude_names=['LAYOUT_EMITTER_NET'])
                        freeze_bn_in_module(self.EMITTER_NET)

                    if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_img_feats_as_input:
                        if not self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.sample_BRDF_feats_instead_of_learn_feats:
                            self.EMITTER_NET_IMG_FEAT_DECODER = self.decoder_to_use(opt, mode=-1, out_channel=self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.img_feats_channels) # same as BRDF decoder; mode==-1 means no activation or postprocessing in the end
                            if self.cfg.MODEL_LAYOUT_EMITTER.emitter.if_freeze:
                                self.turn_off_names(['EMITTER_NET_IMG_FEAT_DECODER'])
                                freeze_bn_in_module(self.EMITTER_NET_IMG_FEAT_DECODER)
                else:
                    # self.LAYOUT_EMITTER_NET_fc = models_layout_emitter.decoder_layout_emitter(opt)
                    pass # already defined  in self.LAYOUT_EMITTER_NET_fc

            if 'ob' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                # object est model from Total3D
                self.OBJECT_NET = models_object_detection.Bdb3DNet(opt)

            if 'mesh' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                self.MESH_NET = models_mesh_reconstruction.DensTMNet(opt)
                
        if self.cfg.MODEL_MATCLS.enable:
            self.MATCLS_NET = model_matcls.netCS(opt=opt, inChannels=4, base_model=resnet.resnet34, if_est_scale=False, if_est_sup = opt.cfg.MODEL_MATCLS.if_est_sup)

        if self.cfg.MODEL_DETECTRON.enable:
            from detectron2.modeling import build_model
            self.DETECTRON = build_model(opt.cfg_detectron)


    def forward(self, input_dict):
        return_dict = {}
        input_dict_guide = None

        if self.cfg.MODEL_MATSEG.enable:
            return_dict_matseg = self.forward_matseg(input_dict) # {'prob': prob, 'embedding': embedding, 'feats_mat_seg_dict': feats_mat_seg_dict}
            input_dict_guide_matseg = return_dict_matseg['feats_matseg_dict']
            input_dict_guide_matseg['guide_from'] = 'matseg'
            input_dict_guide = input_dict_guide_matseg
        else:
            return_dict_matseg = {}
            input_dict_guide_matseg = None
        return_dict.update(return_dict_matseg)

        if self.cfg.MODEL_SEMSEG.enable:
            if self.cfg.MODEL_SEMSEG.if_freeze:
                self.SEMSEG_Net.eval()
            return_dict_semseg = self.forward_semseg(input_dict) # {'prob': prob, 'embedding': embedding, 'feats_mat_seg_dict': feats_mat_seg_dict}

            input_dict_guide_semseg = return_dict_semseg['feats_semseg_dict']
            input_dict_guide_semseg['guide_from'] = 'semseg'
            input_dict_guide = input_dict_guide_semseg
        else:
            return_dict_semseg = {}
            input_dict_guide_semseg = None
        return_dict.update(return_dict_semseg)

        assert not(self.cfg.MODEL_MATSEG.if_guide and self.cfg.MODEL_SEMSEG.if_guide), 'cannot guide from MATSEG and SEMSEG at the same time!'

        if self.cfg.MODEL_GMM.enable:
            input_dict_GMM = input_dict
            print(input_dict_GMM.keys(), input_dict['depthBatch_next'].shape)
            input_dict_GMM['imgs_ref'] = input_dict['im_SDR_RGB'].permute(0, 3, 1, 2)
            input_dict_GMM['imgs_src'] = input_dict['im_SDR_RGB_next'].permute(0, 3, 1, 2)
            input_dict_GMM['dmaps_ref'] = input_dict['depthBatch']
            input_dict_GMM['dmaps_src'] = input_dict['depthBatch_next']
            batch_idx = 0
            self.MODEL_GMM.training_step(input_dict_GMM, batch_idx)

        if self.cfg.MODEL_BRDF.enable:
            if self.cfg.MODEL_BRDF.if_freeze:
                self.BRDF_Net.eval()
            input_dict_extra = {'input_dict_guide': input_dict_guide}
            if (self.cfg.MODEL_MATSEG.if_albedo_pooling and self.cfg.MODEL_MATSEG.albedo_pooling_from == 'pred') \
                or self.cfg.MODEL_MATSEG.use_pred_as_input \
                or self.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv or self.cfg.MODEL_MATSEG.if_albedo_pac_pool or self.cfg.MODEL_MATSEG.if_albedo_pac_conv or self.cfg.MODEL_MATSEG.if_albedo_safenet:
                input_dict_extra.update({'return_dict_matseg': return_dict_matseg})

            return_dict_brdf = self.forward_brdf(input_dict, input_dict_extra=input_dict_extra)
        else:
            return_dict_brdf = {}
        return_dict.update(return_dict_brdf)

        if self.cfg.MODEL_LIGHT.enable:
            if self.cfg.MODEL_LIGHT.if_freeze:
                self.LIGHT_Net.eval()
            return_dict_light = self.forward_light(input_dict, return_dict_brdf=return_dict_brdf)
        else:
            return_dict_light = {}
        return_dict.update(return_dict_light)

        if self.cfg.MODEL_LAYOUT_EMITTER.enable:
            return_dict_layout_emitter = {}

            # --- objects
            if 'ob' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                # self.forward_object_net()
                output_dict = self.OBJECT_NET(input_dict['object_labels']) # {'obj_est_result': {}}
                return_dict_layout_emitter.update(output_dict)

            # --- meshes
            if 'mesh' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                if self.cfg.MODEL_LAYOUT_EMITTER.mesh.loss == 'SVRLoss': # total3d: models/mgnet/modules/network.py L41
                    mesh_output = self.MESH_NET(input_dict['mesh_labels'])
                elif self.cfg.MODEL_LAYOUT_EMITTER.mesh.loss == 'ReconLoss':
                    # output_dict = self.MESH_NET(input_dict['mesh_labels'])
                    # if input_dict['mesh-labels']['mask_flag'] == 1:
                    #     mesh_output = self.mesh_reconstruction(input_dict['mesh-labels']['patch_for_mesh'], input_dict['mesh-labels']['cls_codes_for_mesh'])[0][-1]
                    #     # convert to SUNRGBD coordinates
                    #     mesh_output[:, 2, :] *= -1
                    # else:
                    #     mesh_output = None
                    assert False, 'not implemented... yet'
                output_dict = {'mesh_est_result': mesh_output}
                return_dict_layout_emitter.update(output_dict)

            # --- layout / emitters
            if 'lo' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list or 'em' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                if_vanilla_emitter = 'em' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list and not(self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.enable)
                if_layout = 'lo' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list

                # layout w/ wo/ V1 emitters
                if if_layout or if_vanilla_emitter:
                    if self.cfg.MODEL_LAYOUT_EMITTER.layout.if_indept_encoder:
                        x1, x2, x3, x4, x5, x6 = self.LAYOUT_EMITTER_NET_encoder(input_dict['input_batch_brdf'])
                        encoder_outputs = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6}
                    else:
                        encoder_outputs = return_dict_brdf['encoder_outputs']
                    output_dict = self.LAYOUT_EMITTER_NET_fc(input_feats_dict=encoder_outputs)
                    return_dict_layout_emitter.update(output_dict)

                # V2, V3
                if 'em' in self.cfg.MODEL_LAYOUT_EMITTER.enable_list and self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.enable:
                    if self.cfg.MODEL_LAYOUT_EMITTER.emitter.if_use_est_layout:
                        layout_est_result = output_dict['layout_est_result']

                        lo_bdb3D_result, basis_result, coeffs_result, centroid_result = get_layout_bdb_sunrgbd(self.opt.bins_tensor, \
                            layout_est_result['lo_ori_reg_result'], layout_est_result['lo_ori_cls_result'], \
                            layout_est_result['lo_centroid_result'], layout_est_result['lo_coeffs_result'], \
                            if_return_full=True, if_differentiable=self.cfg.MODEL_LAYOUT_EMITTER.emitter.if_differentiable_layout_input)
                        
                        cam_R_result = get_rotation_matix_result(self.opt.bins_tensor,
                            layout_est_result['pitch_cls_result'], layout_est_result['pitch_reg_result'],
                            layout_est_result['roll_cls_result'], layout_est_result['roll_reg_result'], 
                            if_differentiable=self.cfg.MODEL_LAYOUT_EMITTER.emitter.if_differentiable_layout_input)
                        # pre_layout_reindexed = reindex_layout(pre_layout, pre_cam_R)

                        # input_dict.update({'lo_bdb3D_result': lo_bdb3D_result, 'basis_result': basis_result.detach(), 'coeffs_result': coeffs_result, 'centroid_result': centroid_result.detach(), 'cam_R_result': cam_R_result.detach()})

                        input_dict.update({'lo_bdb3D_result': lo_bdb3D_result, 'basis_result': basis_result, 'coeffs_result': coeffs_result, 'centroid_result': centroid_result, 'cam_R_result': cam_R_result})

                    output_dict = self.forward_emitter_lightAccu(input_dict, return_dict_brdf=return_dict_brdf, return_dict_light=return_dict_light)
                    return_dict_layout_emitter.update(output_dict)
        else:
            return_dict_layout_emitter = {}
        # print(return_dict_layout_emitter.keys()) # dict_keys(['layout_est_result', 'emitter_est_result'])
        return_dict.update(return_dict_layout_emitter)
        
        if self.cfg.MODEL_MATCLS.enable:
            return_dict_matcls = self.forward_matcls(input_dict)
            return_dict.update(return_dict_matcls)

        if self.cfg.MODEL_DETECTRON.enable:
            detectron_dict_list = input_dict['detectron_dict_list']
            detectron_output_list_of_dicts = self.DETECTRON(detectron_dict_list)
            return_dict.update({'detectron_output_list_of_dicts': detectron_output_list_of_dicts})

        return return_dict

    def forward_matseg(self, input_dict):
        input_list = [input_dict['im_batch_matseg']]
        if self.cfg.MODEL_MATSEG.use_semseg_as_input:
            input_list.append(input_dict['semseg_label'].float().unsqueeze(1) / float(self.opt.cfg.MODEL_SEMSEG.semseg_classes))

        if self.cfg.MODEL_MATSEG.if_freeze:
            self.MATSEG_Net.eval()
            with torch.no_grad():
                return self.MATSEG_Net(torch.cat(input_list, 1))
        else:
            return self.MATSEG_Net(torch.cat(input_list, 1))

    def forward_semseg(self, input_dict):
        # im_batch_255 = input_dict['im_uint8']
        # im_batch_255_float = input_dict['im_uint8'].float().permute(0, 3, 1, 2)
        # im_batch_255_float = F.pad(im_batch_255_float, (0, 1, 0, 1))
        # # print(im_batch_255_float.shape, im_batch_255_float_padded.shape)
        # # print(torch.max(im_batch_255_float), torch.min(im_batch_255_float), torch.median(im_batch_255_float), im_batch_255_float.shape, self.mean.shape)
        # im_batch_255_float.sub_(self.mean).div_(self.std)
        # # print(torch.max(im_batch_255_float), torch.min(im_batch_255_float), torch.median(im_batch_255_float))
        
        if self.opt.cfg.MODEL_SEMSEG.if_freeze:
            im_batch_semseg = input_dict['im_batch_semseg_fixed']
            self.SEMSEG_Net.eval()
            with torch.no_grad():
                output_dict_PSPNet = self.SEMSEG_Net(im_batch_semseg, input_dict['semseg_label'])
        else:
            im_batch_semseg = input_dict['im_batch_semseg']
            output_dict_PSPNet = self.SEMSEG_Net(im_batch_semseg, input_dict['semseg_label'])

        output_PSPNet = output_dict_PSPNet['x']
        feat_dict_PSPNet = output_dict_PSPNet['feat_dict']
        main_loss, aux_loss = output_dict_PSPNet['main_loss'], output_dict_PSPNet['aux_loss']

        _, _, h_i, w_i = im_batch_semseg.shape
        _, _, h_o, w_o = output_PSPNet.shape
        # print(h_o, h_i, w_o, w_i)
        assert (h_o == h_i) and (w_o == w_i)
        # if (h_o != h_i) or (w_o != w_i):
            # output_PSPNet = F.interpolate(output_PSPNet, (h_i, w_i), mode='bilinear', align_corners=True)
        # output_PSPNet = output_PSPNet[:, :, :-1, :-1]
        # output_PSPNet_softmax = F.softmax(output_PSPNet, dim=1)

        return_dict = {'semseg_pred': output_PSPNet, 'PSPNet_main_loss': main_loss, 'PSPNet_aux_loss': aux_loss}
        return_dict.update({'feats_semseg_dict': feat_dict_PSPNet})

        return return_dict

    def forward_brdf(self, input_dict, input_dict_extra={}):
        assert 'input_dict_guide' in input_dict_extra
        if 'input_dict_guide' in input_dict_extra:
            input_dict_guide = input_dict_extra['input_dict_guide']
        else:
            input_dict_guide = None

        input_list = [input_dict['input_batch_brdf']]

        if self.opt.cfg.MODEL_SEMSEG.use_as_input:
            input_list.append(input_dict['semseg_label'].float().unsqueeze(1) / float(self.opt.cfg.MODEL_SEMSEG.semseg_classes))
        if self.opt.cfg.MODEL_MATSEG.use_as_input:
            input_list.append(input_dict['matAggreMapBatch'].float() / float(255.))
        if self.opt.cfg.MODEL_MATSEG.use_pred_as_input:
            matseg_logits = input_dict_extra['return_dict_matseg']['logit']
            matseg_embeddings = input_dict_extra['return_dict_matseg']['embedding']
            mat_notlight_mask_cpu = input_dict['mat_notlight_mask_cpu']
            _, _, predict_segmentation = logit_embedding_to_instance(mat_notlight_mask_cpu, matseg_logits, matseg_embeddings, self.opt)
            input_list.append(predict_segmentation.float().unsqueeze(1) / float(self.opt.cfg.MODEL_SEMSEG.semseg_classes))
        
        input_tensor = torch.cat(input_list, 1)
        #     # a = input_dict['semseg_label'].float().unsqueeze(1) / float(self.opt.cfg.MODEL_SEMSEG.semseg_classes)
        #     # print(torch.max(a), torch.min(a), torch.median(a))
        #     # print('--', torch.max(input_dict['input_batch_brdf']), torch.min(input_dict['input_batch_brdf']), torch.median(input_dict['input_batch_brdf']))
        # else:
        #     input_tensor = input_dict['input_batch_brdf']
        x1, x2, x3, x4, x5, x6 = self.BRDF_Net['encoder'](input_tensor)

        return_dict = {'encoder_outputs': {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6}}
        albedo_output = {}

        if self.cfg.MODEL_BRDF.enable_BRDF_decoders:
            input_extra_dict = {}
            if input_dict_guide is not None:
                input_extra_dict.update({'input_dict_guide': input_dict_guide})
            if self.cfg.MODEL_MATSEG.if_albedo_pooling:
                input_extra_dict.update({'matseg-instance': input_dict['instance'], 'semseg-num_mat_masks_batch': input_dict['num_mat_masks_batch']})
                input_extra_dict.update({'im_trainval_RGB': input_dict['im_trainval_RGB']})
                if self.cfg.MODEL_MATSEG.albedo_pooling_from == 'pred':
                    assert input_dict_extra is not None
                    assert input_dict_extra['return_dict_matseg'] is not None
                    input_extra_dict.update({'matseg-logits': input_dict_extra['return_dict_matseg']['logit'], 'matseg-embeddings': input_dict_extra['return_dict_matseg']['embedding'], \
                        'mat_notlight_mask_cpu': input_dict['mat_notlight_mask_cpu']})
            if self.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv or self.cfg.MODEL_MATSEG.if_albedo_pac_pool or self.cfg.MODEL_MATSEG.if_albedo_pac_conv or self.cfg.MODEL_MATSEG.if_albedo_safenet:
                assert input_dict_extra is not None
                assert input_dict_extra['return_dict_matseg'] is not None
                input_extra_dict.update({'im_trainval_RGB': input_dict['im_trainval_RGB'], 'mat_notlight_mask_gpu_float': input_dict['mat_notlight_mask_gpu_float']})
                input_extra_dict.update({'matseg-embeddings': input_dict_extra['return_dict_matseg']['embedding']})

            if 'al' in self.cfg.MODEL_BRDF.enable_list:
                albedo_output = self.BRDF_Net['albedoDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_extra_dict=input_extra_dict)
                albedoPred = 0.5 * (albedo_output['x_out'] + 1)
                if (not self.opt.cfg.DATASET.if_no_gt):
                    input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
                    if not self.cfg.MODEL_BRDF.use_scale_aware_albedo:
                        albedoPred = models_brdf.LSregress(albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred),
                                input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoPred)
                albedoPred = torch.clamp(albedoPred, 0, 1)
                return_dict.update({'albedoPred': albedoPred})
            if 'no' in self.cfg.MODEL_BRDF.enable_list:
                normalPred = self.BRDF_Net['normalDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_extra_dict=input_extra_dict)['x_out']
                return_dict.update({'normalPred': normalPred})
            if 'ro' in self.cfg.MODEL_BRDF.enable_list:
                roughPred = self.BRDF_Net['roughDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_extra_dict=input_extra_dict)['x_out']
                return_dict.update({'roughPred': roughPred})
            if 'de' in self.cfg.MODEL_BRDF.enable_list:
                if not self.cfg.MODEL_BRDF.use_scale_aware_depth and (not self.opt.cfg.DATASET.if_no_gt):
                    depthPred = 0.5 * (self.BRDF_Net['depthDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_extra_dict=input_extra_dict)['x_out'] + 1) # [-1, 1] -> [0, 2] -> [0, 1]
                    depthPred = models_brdf.LSregress(depthPred *  input_dict['segAllBatch'].expand_as(depthPred),
                            input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthPred)
                else:
                    depthPred = self.BRDF_Net['depthDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_extra_dict=input_extra_dict)['x_out']

                return_dict.update({'depthPred': depthPred})


            # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape)
            # return_dict.update({'albedoPred': albedosPred, 'normalPred': normalPred, 'roughPred': roughPred, 'depthPred': depthPred})

        if self.cfg.MODEL_BRDF.enable_semseg_decoder:
            semsegPred = self.BRDF_Net['semsegDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6)['x_out']
            return_dict.update({'semseg_pred': semsegPred})
            
        if self.cfg.MODEL_MATSEG.if_albedo_pooling or self.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv or self.cfg.MODEL_MATSEG.if_albedo_pac_pool or self.cfg.MODEL_MATSEG.if_albedo_safenet:
            return_dict.update({'im_trainval_RGB_mask_pooled_mean': albedo_output['im_trainval_RGB_mask_pooled_mean']})
            if 'kernel_list' in albedo_output:
                return_dict.update({'kernel_list': albedo_output['kernel_list']})
            if 'embeddings' in albedo_output:
                return_dict.update({'embeddings': albedo_output['embeddings']})
            if 'affinity' in albedo_output:
                return_dict.update({'affinity': albedo_output['affinity'], 'sample_ij': albedo_output['sample_ij']})
            

        return return_dict

    def forward_emitter_lightAccu(self, input_dict, return_dict_brdf={}, return_dict_light={}):
        if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_GT_light:
            envmapsBatch = input_dict['envmapsBatch']
        else:
            # envmapsBatch = return_dict_light['envmapsPredImage']
            envmapsBatch = return_dict_light['envmapsPredScaledImage'] # should not assume we have this because this aligns with GT for scale

        if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_GT_brdf:
            normalBatch = input_dict['normalBatch']
            depthBatch = input_dict['depthBatch'].squeeze(1)
        else:
            normalBatch = return_dict_brdf['normalPred']
            depthBatch = return_dict_brdf['depthPred'].squeeze(1)

        # cam_K_batch = input_dict['layout_labels']['cam_K']
        cam_K_batch = input_dict['layout_labels']['cam_K_scaled']
        if self.opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_use_est_layout:
            cam_R_batch = input_dict['cam_R_result']
            layout_batch = input_dict['lo_bdb3D_result']
            basis_batch, coeffs_batch, centroid_batch = input_dict['basis_result'], input_dict['coeffs_result'], input_dict['centroid_result']
        else:
            cam_R_batch = input_dict['layout_labels']['cam_R_gt']
            if self.opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_train_with_reindexed_layout:
                layout_batch = input_dict['layout_labels']['lo_bdb3D_reindexed']
            else:
                layout_batch = input_dict['layout_labels']['lo_bdb3D']
            # basis_batch, coeffs_batch, centroid_batch = {key: gt_dict_lo['lo_bdb3D_full'][key][sample_idx].detach().cpu().numpy() for key in gt_dict_lo['lo_bdb3D_full']}
            basis_batch = input_dict['layout_labels']['lo_bdb3D_full']['basis'].cuda().float()
            coeffs_batch = input_dict['layout_labels']['lo_bdb3D_full']['coeffs'].cuda().float()
            centroid_batch = input_dict['layout_labels']['lo_bdb3D_full']['centroid'].cuda().float()


        # print(depthBatch.shape, normalBatch.shape, envmapsBatch.shape, cam_K_batch.shape, cam_R_batch.shape, layout_batch.shape)

        input_dict_light_accu = {'normalPred_lightAccu':normalBatch, 'depthPred_lightAccu': depthBatch, 'envmapsPredImage_lightAccu': envmapsBatch, 'cam_K': cam_K_batch, 'cam_R': cam_R_batch, \
            'layout': layout_batch, 'basis': basis_batch, 'coeffs': coeffs_batch, 'centroid': centroid_batch, }
        return_dict_layout_emitter = {'emitter_input': {}, 'emitter_misc': {}}

        # ---- accu lights
        return_dict_lightAccu = self.EMITTER_LIGHT_ACCU_NET(input_dict_light_accu)
        envmap_lightAccu_mean = return_dict_lightAccu['envmap_lightAccu_mean'].view(-1, 6, 8, 8, 3).permute(0, 1, 4, 2, 3)

        if self.cfg.MODEL_LIGHT.if_transform_to_LightNet_coords:
            camx, camy, normalPred = return_dict_lightAccu['camx'], return_dict_lightAccu['camy'], return_dict_lightAccu['normalPred'] # all: torch.Size([B, 3, 120, 160])
            # print(camx.shape, camy.shape, normalPred.shape)
            axisPred, lambPred, weightPred = return_dict_light['LightNet_preds']['axisPred'], return_dict_light['LightNet_preds']['lambPred'], return_dict_light['LightNet_preds']['weightPred']
            segEnvBatch = return_dict_light['LightNet_preds']['segEnvBatch']
            axisPred_LightNetCoords = axisPred[:, :, 0:1, :, :] * camx.unsqueeze(1) + axisPred[:, :, 1:2, :, :] * camy.unsqueeze(1) + axisPred[:, :, 2:3, :, :] * normalPred.unsqueeze(1) # [B, 12, 3, 120, 160]
            assert axisPred.shape == axisPred_LightNetCoords.shape
            envmapsPredImage, _, _, _ = self.non_learnable_layers['output2env'].output2env(axisPred_LightNetCoords, lambPred, weightPred )
            envmapsPredScaledImage = models_brdf.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                input_dict['envmapsBatch'] * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage )
            return_dict_layout_emitter['emitter_misc']['envmapsPredScaledImage_LightNetCoords'] = envmapsPredScaledImage
            return_dict_layout_emitter['emitter_misc']['envmapsPred_LightNetCoords'] = envmapsPredImage
            return_dict_layout_emitter['emitter_misc']['LightNet_misc'] = {
                'axisPred': axisPred, 
                'lambPred': lambPred, 
                'weightPred': weightPred, 
                'camx': camx, 
                'camy': camy, 
                'normalPred': normalPred, 
                'segEnvBatch': segEnvBatch, 
                'axisPred_LightNetCoords': axisPred_LightNetCoords
            }

            
        grid_size = self.EMITTER_LIGHT_ACCU_NET.grid_size
        im_height, im_width = self.opt.cfg.DATA.im_height, self.opt.cfg.DATA.im_width

        V2_input_list = [envmap_lightAccu_mean]

        # ---- sample image features using reproj cell centers
        if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_img_feats_as_input:
            # ---- get img feats
            # x1 torch.Size([2, 64, 120, 160])
            # x2 torch.Size([2, 128, 60, 80])
            # x3 torch.Size([2, 256, 30, 40])
            # x4 torch.Size([2, 256, 15, 20])
            # x5 torch.Size([2, 512, 7, 10])
            # x6 torch.Size([2, 1024, 7, 10])
            x1, x2, x3, x4, x5, x6 = return_dict_brdf['encoder_outputs']['x1'], return_dict_brdf['encoder_outputs']['x2'], return_dict_brdf['encoder_outputs']['x3'], \
                return_dict_brdf['encoder_outputs']['x4'], return_dict_brdf['encoder_outputs']['x5'], return_dict_brdf['encoder_outputs']['x6']
            if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.sample_BRDF_feats_instead_of_learn_feats:
                x2_up = F.interpolate(x2, scale_factor=2, mode='bilinear')
                x3_up = F.interpolate(x3, scale_factor=4, mode='bilinear')
                x4_up = F.interpolate(x4, scale_factor=8, mode='bilinear')
                x1234_feat_concat = torch.cat([x1, x2_up, x3_up, x4_up], axis=1)
                # print(x1234_feat_concat.shape)
                img_feat_output = F.interpolate(x1234_feat_concat, scale_factor=2, mode='bilinear')
                # print(x1234_feat_concat.shape)
                feat_map = img_feat_output.unsqueeze(2) # [B, D, 1, H, W]
            else:
                img_feat_output = self.EMITTER_NET_IMG_FEAT_DECODER(input_dict['imBatch'], x1, x2, x3, x4, x5, x6)['x_out'] # [B, 8, 240, 320]
                feat_map = torch.cat([img_feat_output, input_dict['imBatch']], dim=1).unsqueeze(2) # [B, D+3, 1, H, W]
            assert img_feat_output.shape[2:]==(im_height, im_width)


            # ---- get reprojected cell centers
            verts_all_Total3D = return_dict_lightAccu['verts_all_Total3D'] # [B, 6, 8, 8, 3, 4]
            verts_center_all_Total3D = verts_all_Total3D.mean(-1) # [B, 6, 8, 8, 3]

            cam_K_scaled_batch = input_dict['layout_labels']['cam_K_scaled']
            cam_dict = {'origin': torch.tensor([0., 0., 0.]).cuda(), 'cam_K': cam_K_scaled_batch}
            cam_axes_batch = cam_R_batch

            simpleSceneTorch = SimpleSceneTorchBatch(cam_dict, im_height=im_height, im_width=im_width)
            simpleSceneTorch.form_camera(cam_axes_batch)
            
            batch_size = verts_center_all_Total3D.shape[0]
            # -> [B, 384, 2] torch.float32, [B, 384] torch.bool
            verts_proj, front_flags = simpleSceneTorch.transform_and_proj(verts_center_all_Total3D.reshape(batch_size, -1, 3)) # assuming align_corners=False, meaning the upper-left corner is 0 and lower corner is (W, H)
            verts_proj[torch.logical_not(front_flags.unsqueeze(-1).repeat(1, 1, 2))] = -100

            feat_height, feat_width = feat_map.shape[-2:]
            verts_proj_reshape = verts_proj.view(batch_size, 6, grid_size, grid_size, 2) # [B, 6, 8, 8, 2]
            wh_tensor = torch.tensor([feat_width, feat_height]).reshape(1, 1, 1, 1, 2).cuda().float()
            verts_proj_reshape_normalized = verts_proj_reshape/wh_tensor * 2. - 1.
            verts_proj_reshape_normalized_concat = torch.cat([verts_proj_reshape_normalized, torch.zeros(batch_size, 6, grid_size, grid_size, 1).cuda()], -1) # [B, 6, 8, 8, 3]

            # [B, D, 1, H, W], [B, 6, 8, 8, 3] -> [B, D, 6, 8, 8]
            img_feat_map_sampled = torch.nn.functional.grid_sample(feat_map, verts_proj_reshape_normalized_concat, padding_mode="zeros", align_corners=False)
            return_dict_layout_emitter['emitter_input']['img_feat_map_sampled'] = img_feat_map_sampled

            V2_input_list.append(img_feat_map_sampled.transpose(1, 2)) # [B, D, 6, 8, 8] -> [B, 6, D, 8, 8]


        if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.version in ['V1', 'V2']:
            V2_input_tensor = torch.cat(V2_input_list, 2) #  # [B, 6, D(3), H(grid_size), W(grid_size)]
            # ic([a.shape for a in V2_input_list])
            emitter_net_outputs = self.EMITTER_NET(V2_input_tensor)
            return_dict_layout_emitter.update(emitter_net_outputs)

        elif self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.version in ['V3']:
            # Scatter lights to each emitter
            scattered_light, emitter_global2localLightNet_trans_matrix = self.EMITTER_LIGHT_ACCU_NET.scatter_light_to_hemisphere(return_dict_lightAccu, if_scatter=True)
            
            # ---- Get params to transform cell_axis from LightNet coords -> Total3D coords
            Total3D_to_LightNet_transform_params = return_dict_lightAccu['Total3D_to_LightNet_transform_params']
            cam_R_transform_matrix_pre, post_transform_matrix = Total3D_to_LightNet_transform_params['cam_R_transform_matrix_pre'], Total3D_to_LightNet_transform_params['post_transform_matrix']
            inv_post_transform_matrix_expand = post_transform_matrix.unsqueeze(1).unsqueeze(1).unsqueeze(1).transpose(-1, -2)
            inv_inv_cam_R_transform_matrix_pre_expand = cam_R_transform_matrix_pre.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            transform_params_LightNet2Total3D = {'inv_post_transform_matrix_expand': inv_post_transform_matrix_expand, 'inv_inv_cam_R_transform_matrix_pre_expand': inv_inv_cam_R_transform_matrix_pre_expand}

            T_LightNet2Total3D_rightmult = transform_params_LightNet2Total3D['inv_post_transform_matrix_expand'] @ transform_params_LightNet2Total3D['inv_inv_cam_R_transform_matrix_pre_expand']

            # ---- Get the dir meshgrid outside (abs/relative) for all cells
            emitter_outdirs_meshgrid_global_lightNet = self.EMITTER_LIGHT_ACCU_NET.get_emitter_outdirs_meshgrid(emitter_global2localLightNet_trans_matrix)
            normal_outside_lightNet = - return_dict_lightAccu['normal_array_lightNet'].unsqueeze(2).unsqueeze(2).unsqueeze(2) # [B, 6, 1, 1, 1, 3]
            normal_outside_lightNet = normal_outside_lightNet.repeat(1, 1, grid_size**2, 1, 1, 1) # normal inside --negative--> normal outside # [B, ngrids(6*8*8, 1, 1, 3]
            normal_outside_lightNet = normal_outside_lightNet.view(normal_outside_lightNet.shape[0], 6*grid_size*grid_size, 1, 1, 3) # [B, ngrids, 1, 1, 3]

            emitter_outdirs_meshgrid_global_lightNet_outside = - emitter_outdirs_meshgrid_global_lightNet # flip the hemisphere dirs so that they point to the outside of the wall
            
            # ---- LightNet -> Total3D (dirs & normal outside)
            emitter_outdirs_meshgrid_Total3D_outside_abs = (emitter_outdirs_meshgrid_global_lightNet_outside.unsqueeze(-2) @ T_LightNet2Total3D_rightmult).squeeze(-2) # [B, 384, 8, 16, 3]
            normal_outside_Total3D = (normal_outside_lightNet.unsqueeze(-2) @ T_LightNet2Total3D_rightmult).squeeze(-2) # [B, 384, 8, 16, 3]            
            
            if self.cfg.MODEL_LAYOUT_EMITTER.emitter.relative_dir:
                emitter_outdirs_meshgrid_Total3D_outside = emitter_outdirs_meshgrid_Total3D_outside_abs - normal_outside_Total3D
            else:
                emitter_outdirs_meshgrid_Total3D_outside = emitter_outdirs_meshgrid_Total3D_outside_abs

            # ---- sample env map!
            if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_envmap_as_input:
                # -- if not scatter: use_sampled_envmap_as_input
                input_dict_sampler = {'emitter_outdirs_meshgrid_Total3D_outside': emitter_outdirs_meshgrid_Total3D_outside_abs, \
                     'transform_R_RAW2Total3D': input_dict['emitter_labels']['transform_R_RAW2Total3D'], \
                     'im_envmap_ori': input_dict['emitter_labels']['im_envmap_ori']}

                im_envmap_sampled = self.EMITTER_LIGHT_ACCU_NET.sample_envmap(input_dict_sampler) # [B, 3, 384, 8, 16]
                im_envmap_sampled = im_envmap_sampled.permute(0, 2, 3, 4, 1) # [B, 384, 8, 16, 3]
                window_mask = (input_dict['emitter_labels']['cell_cls'] == 1).float() # [B, 6, 8, 8]
                window_mask = window_mask.view(window_mask.shape[0], -1) # [B, 384]
                window_mask = window_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # print(window_mask.shape, im_envmap_sampled.shape, scattered_light.shape)
                scattered_light = im_envmap_sampled * window_mask + scattered_light * (1-window_mask)

            # ---- run the emitter net!
            input_dict_lightAccuScatter_UNet_V3 = {'scattered_light': scattered_light, 'emitter_outdirs_meshgrid_Total3D_outside': emitter_outdirs_meshgrid_Total3D_outside}
            if self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_img_feats_as_input:
                input_dict_lightAccuScatter_UNet_V3.update({'img_feat_map_sampled': img_feat_map_sampled})

            emitter_net_outputs = self.EMITTER_NET(input_dict_lightAccuScatter_UNet_V3)

            # ---- gather outputs
            return_dict_layout_emitter.update(emitter_net_outputs)
            return_dict_layout_emitter['emitter_est_result'].update({'scattered_light': scattered_light})

            return_dict_layout_emitter['emitter_est_result'].update({'envmap_lightAccu': return_dict_lightAccu['envmap_lightAccu'], \
                'emitter_outdirs_meshgrid_Total3D_outside': emitter_outdirs_meshgrid_Total3D_outside, 'normal_outside_Total3D': normal_outside_Total3D, \
                'points_backproj': return_dict_lightAccu['points_backproj'], 'depthPred': return_dict_lightAccu['depthPred'], 'T_LightNet2Total3D_rightmult': T_LightNet2Total3D_rightmult})
        else:
            raise ValueError('Invalid self.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.version')


        return_dict_layout_emitter['emitter_est_result'].update({'envmap_lightAccu_mean': envmap_lightAccu_mean, 'points_sampled_mask_expanded': return_dict_lightAccu['points_sampled_mask_expanded'], \
            'emitter_outdirs_meshgrid_Total3D_outside_abs': emitter_outdirs_meshgrid_Total3D_outside_abs, 'normal_outside_Total3D': normal_outside_Total3D})

        return_dict_layout_emitter['emitter_input'].update({'normalPred_lightAccu':normalBatch, 'depthPred_lightAccu': depthBatch, 'envmapsPredImage_lightAccu': envmapsBatch})

        return return_dict_layout_emitter

    def forward_light(self, input_dict, return_dict_brdf):
        # Normalize Albedo and depth
        if not self.cfg.MODEL_LIGHT.use_GT_brdf:
            depthPred = return_dict_brdf['depthPred']
            albedoPred = return_dict_brdf['albedoPred']
            normalPred = return_dict_brdf['normalPred']
            roughPred = return_dict_brdf['roughPred']
        else:
            albedoPred = input_dict['albedoBatch']
            depthPred = input_dict['depthBatch']
            normalPred = input_dict['normalBatch']
            roughPred = input_dict['roughBatch']

        if self.cfg.MODEL_LIGHT.freeze_BRDF_Net and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            assert self.BRDF_Net.training == False
            
        # note: normalization/rescaling also needed for GT BRDFs
        bn, ch, nrow, ncol = albedoPred.size()
        albedoPred = albedoPred.view(bn, -1)
        albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        albedoPred = albedoPred.view(bn, ch, nrow, ncol)

        bn, ch, nrow, ncol = depthPred.size()
        depthPred = depthPred.view(bn, -1)
        depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        depthPred = depthPred.view(bn, ch, nrow, ncol)

        imBatchLarge = F.interpolate(input_dict['imBatch'], [480, 640], mode='bilinear')
        albedoPredLarge = F.interpolate(albedoPred, [480, 640], mode='bilinear')
        depthPredLarge = F.interpolate(depthPred, [480, 640], mode='bilinear')
        normalPredLarge = F.interpolate(normalPred, [480, 640], mode='bilinear')
        roughPredLarge = F.interpolate(roughPred, [480,640], mode='bilinear')

        input_batch = torch.cat([imBatchLarge, albedoPredLarge,
            0.5*(normalPredLarge+1), 0.5 * (roughPredLarge+1), depthPredLarge ], dim=1 )

        if self.opt.cascadeLevel == 0:
            x1, x2, x3, x4, x5, x6 = self.LIGHT_Net['lightEncoder'](input_batch.detach() )
        else:
            assert self.opt.cascadeLevel > 0
            x1, x2, x3, x4, x5, x6 = self.LIGHT_Net['lightEncoder'](input_batch.detach(), input_dict['envmapsPreBatch'].detach() )

        # Prediction
        axisPred_ori = self.LIGHT_Net['axisDecoder'](x1, x2, x3, x4, x5, x6, input_dict['envmapsBatch'] )
        lambPred_ori = self.LIGHT_Net['lambDecoder'](x1, x2, x3, x4, x5, x6, input_dict['envmapsBatch'] )
        weightPred_ori = self.LIGHT_Net['weightDecoder'](x1, x2, x3, x4, x5, x6, input_dict['envmapsBatch'] )
        bn, SGNum, _, envRow, envCol = axisPred_ori.size()
        # envmapsPred = torch.cat([axisPred.view(bn, SGNum * 3, envRow, envCol ), lambPred, weightPred], dim=1)

        imBatchSmall = F.adaptive_avg_pool2d(input_dict['imBatch'], (self.cfg.MODEL_LIGHT.envRow, self.cfg.MODEL_LIGHT.envCol) )
        segBatchSmall = F.adaptive_avg_pool2d(input_dict['segBRDFBatch'], (self.cfg.MODEL_LIGHT.envRow, self.cfg.MODEL_LIGHT.envCol) )
        notDarkEnv = (torch.mean(torch.mean(torch.mean(input_dict['envmapsBatch'], 4), 4), 1, True ) > 0.001 ).float()
        segEnvBatch = (segBatchSmall * input_dict['envmapsIndBatch'].expand_as(segBatchSmall) ).unsqueeze(-1).unsqueeze(-1)
        segEnvBatch = segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)
        
        return_dict = {}

        # Compute the recontructed error
        envmapsPredImage, axisPred, lambPred, weightPred = self.non_learnable_layers['output2env'].output2env(axisPred_ori, lambPred_ori, weightPred_ori )

        pixelNum_recon = max( (torch.sum(segEnvBatch ).cpu().data).item(), 1e-5)
        envmapsPredScaledImage = models_brdf.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
            input_dict['envmapsBatch'] * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage )
        # ic(torch.max(input_dict['envmapsBatch']), torch.min(input_dict['envmapsBatch']),torch.median(input_dict['envmapsBatch']))
        # ic(torch.max(envmapsPredImage), torch.min(envmapsPredImage),torch.median(envmapsPredImage))
        # ic(torch.max(envmapsPredScaledImage), torch.min(envmapsPredScaledImage),torch.median(envmapsPredScaledImage))

        return_dict.update({'envmapsPredImage': envmapsPredImage, 'envmapsPredScaledImage': envmapsPredScaledImage, 'segEnvBatch': segEnvBatch, \
            'imBatchSmall': imBatchSmall, 'segBatchSmall': segBatchSmall, 'pixelNum_recon': pixelNum_recon}) 

        # Compute the rendered error
        pixelNum_render = max( (torch.sum(segBatchSmall ).cpu().data).item(), 1e-5 )
        
        if not self.cfg.MODEL_LIGHT.use_GT_brdf:
            normal_input, rough_input = return_dict_brdf['normalPred'], return_dict_brdf['roughPred']
        else:
            normal_input, rough_input = input_dict['normalBatch'], input_dict['roughBatch']

        if self.cfg.MODEL_LIGHT.use_GT_light:
            envmapsImage_input = input_dict['envmapsBatch']
        else:
            envmapsImage_input = envmapsPredImage

        diffusePred, specularPred = self.non_learnable_layers['renderLayer'].forwardEnv(normalPred=normal_input, envmap=envmapsImage_input, diffusePred=albedoPred.detach(), roughPred=rough_input)

        diffusePredScaled, specularPredScaled = models_brdf.LSregressDiffSpec(
            diffusePred.detach(),
            specularPred.detach(),
            imBatchSmall,
            diffusePred, specularPred )

        renderedImPred_hdr = diffusePredScaled + specularPredScaled
        renderedImPred = torch.clamp(renderedImPred_hdr, 0, 1)
        renderedImPred_sdr = torch.clamp(renderedImPred_hdr ** (1.0/2.2), 0, 1)

        return_dict.update({'renderedImPred': renderedImPred, 'renderedImPred_sdr': renderedImPred_sdr, 'pixelNum_render': pixelNum_render}) 
        return_dict.update({'LightNet_preds': {'axisPred': axisPred_ori, 'lambPred': lambPred_ori, 'weightPred': weightPred_ori, 'segEnvBatch': segEnvBatch}})

        return return_dict
    
    def forward_matcls(self, input_dict):
        input_batch = torch.cat([input_dict['imBatch'], input_dict['mat_mask_batch'].to(torch.float32)], 1)
        output = self.MATCLS_NET(input_batch)
        _, matcls_argmax = torch.max(output['material'], 1)
        return_dict = {'matcls_output': output['material'], 'matcls_argmax': matcls_argmax}
        if self.opt.cfg.MODEL_MATCLS.if_est_sup:
            _, matcls_sup_argmax = torch.max(output['material_sup'], 1)
            return_dict.update({'matcls_sup_output': output['material_sup'], 'matcls_sup_argmax': matcls_sup_argmax})
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

    def load_pretrained_MODEL_BRDF(self, pretrained_pth_name='check_cascade0_w320_h240'):
        if self.opt.if_cluster:
            pretrained_path = '/viscompfs/users/ruizhu/models_ckpt/' + pretrained_pth_name
        else:
            pretrained_path = '/home/ruizhu/Documents/Projects/semanticInverse/models_ckpt/' + pretrained_pth_name
        loaded_strings = []
        for saved_name in ['encoder', 'albedo', 'normal', 'rough', 'depth']:
            if saved_name == 'encoder':
                module_name = saved_name
            else:
                module_name = saved_name+'Decoder'
            # pickle_path = '{0}/{1}{2}_{3}.pth'.format(pretrained_path, saved_name, cascadeLevel, epochIdFineTune) 
            pickle_path = pretrained_path % saved_name
            # print('----- Loading %s  for module %s'%(pickle_path, module_name))
            # print(self.opt.cfg.MODEL_BRDF.enable_list, self.cfg.MODEL_BRDF.enable_BRDF_decoders)
            # self.print_net()
            self.BRDF_Net[module_name].load_state_dict(
                torch.load(pickle_path).state_dict())
            loaded_strings.append(saved_name)

        self.logger.info(magenta('Loaded pretrained BRDF from %s: %s'%(pretrained_pth_name, '+'.join(loaded_strings))))
    
    def load_pretrained_MODEL_LIGHT(self, pretrained_pth_name='check_cascadeLight0_sg12_offset1.0'):
        if self.opt.if_cluster:
            pretrained_path = '/viscompfs/users/ruizhu/models_ckpt/' + pretrained_pth_name
        else:
            pretrained_path = '/home/ruizhu/Documents/Projects/semanticInverse/models_ckpt/' + pretrained_pth_name
        loaded_strings = []
        for saved_name in ['lightEncoder', 'axisDecoder', 'lambDecoder', 'weightDecoder', ]:
            # pickle_path = '{0}/{1}{2}_{3}.pth'.format(pretrained_path, saved_name, cascadeLevel, epochIdFineTune) 
            pickle_path = pretrained_path % saved_name
            print('Loading ' + pickle_path)
            self.LIGHT_Net[saved_name].load_state_dict(
                torch.load(pickle_path).state_dict())
            loaded_strings.append(saved_name)

        self.logger.info(magenta('Loaded pretrained LIGHT from %s: %s'%(pretrained_pth_name, '+'.join(loaded_strings))))

    def load_pretrained_semseg(self):
        # self.print_net()
        model_path = os.path.join(self.semseg_path, self.opt.cfg.MODEL_SEMSEG.pretrained_pth)
        if os.path.isfile(model_path):
            self.logger.info(red("=> loading checkpoint '{}'".format(model_path)))
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))['state_dict']
            # print(state_dict.keys())
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            replace_dict = {'layer0.0': 'layer0_1.0', 'layer0.1': 'layer0_1.1', 'layer0.3': 'layer0_2.0', 'layer0.4': 'layer0_2.1', 'layer0.6': 'layer0_3.0', 'layer0.7': 'layer0_3.1'}
            state_dict = {k.replace(key, replace_dict[key]): v for k, v in state_dict.items() for key in replace_dict}
            
            self.SEMSEG_Net.load_state_dict(state_dict, strict=True)
            self.logger.info(red("=> loaded checkpoint '{}'".format(model_path)))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))

    def load_pretrained_matseg(self):
        # self.print_net()
        model_path = os.path.join(self.opt.CKPT_PATH, self.opt.cfg.MODEL_MATSEG.pretrained_pth)
        if os.path.isfile(model_path):
            self.logger.info(red("=> loading checkpoint '{}'".format(model_path)))
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))['model']
            # print(state_dict.keys())
            state_dict = {k.replace('UNet.', '').replace('MATSEG_Net.', ''): v for k, v in state_dict.items()}
            state_dict = {k: v for k, v in state_dict.items() if ('pred_depth' not in k) and ('pred_surface_normal' not in k) and ('pred_param' not in k)}

            # replace_dict = {'layer0.0': 'layer0_1.0', 'layer0.1': 'layer0_1.1', 'layer0.3': 'layer0_2.0', 'layer0.4': 'layer0_2.1', 'layer0.6': 'layer0_3.0', 'layer0.7': 'layer0_3.1'}
            # state_dict = {k.replace(key, replace_dict[key]): v for k, v in state_dict.items() for key in replace_dict}
            
            # print(state_dict.keys())
            self.MATSEG_Net.load_state_dict(state_dict, strict=True)
            self.logger.info(red("=> loaded checkpoint '{}'".format(model_path)))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))

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

# class guideNet(nn.Module):
#     def __init__(self, opt):
#         super(guideNet, self).__init__()
#         self.opt = opt
#         self.guide_C = self.opt.cfg.MODEL_MATSEG.guide_channels
#         self.relu = nn.ReLU(inplace=True)

#     #     self.process_convs = nn.ModuleDict(
#     #         {
#     #             'p0': nn.Sequential(nn.Conv2d(64, self.guide_C, (3, 3), padding=1), nn.Conv2d(self.guide_C, self.guide_C, (1, 1)), self.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p1': nn.Sequential(nn.Conv2d(64, self.guide_C, (3, 3), padding=1), nn.Conv2d(self.guide_C, self.guide_C, (1, 1)), self.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p2': nn.Sequential(nn.Conv2d(64, self.guide_C, (3, 3), padding=1), nn.Conv2d(self.guide_C, self.guide_C, (1, 1)), self.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p3': nn.Sequential(nn.Conv2d(64, self.guide_C, (3, 3), padding=1), nn.Conv2d(self.guide_C, self.guide_C, (1, 1)), self.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p4': nn.Sequential(nn.Conv2d(64, self.guide_C, (3, 3), padding=1), nn.Conv2d(self.guide_C, self.guide_C, (1, 1)), self.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p5': nn.Sequential(nn.Conv2d(64, self.guide_C, (3, 3), padding=1), nn.Conv2d(self.guide_C, self.guide_C, (1, 1)), self.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #         }
#     #    )
#     #     self.process_convs = nn.ModuleDict(
#     #         {
#     #             'p0': nn.Sequential(nn.Conv2d(64, self.guide_Cself.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p1': nn.Sequential(nn.Conv2d(64, self.guide_Cself.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p2': nn.Sequential(nn.Conv2d(64, self.guide_Cself.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p3': nn.Sequential(nn.Conv2d(64, self.guide_Cself.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p4': nn.Sequential(nn.Conv2d(64, self.guide_Cself.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #             'p5': nn.Sequential(nn.Conv2d(64, self.guide_Cself.relu, nn.GroupNorm(num_groups=4, num_channels=self.guide_C)), 
#     #         }
#     #    )

#     def forward(self, feats_mat_seg_dict):
#         feats_mat_seg_dict_processed = {}
#         for key in feats_mat_seg_dict:
#             feats = feats_mat_seg_dict[key]
#             # processed_feats = self.process_convs[key](feats)
#             # feats_mat_seg_dict_processed.update({key: processed_feats})
#             feats_mat_seg_dict_processed.update({key: feats})
            
#             # p0 torch.Size([16, 64, 192, 256]) torch.Size([16, 64, 192, 256])
#             # p1 torch.Size([16, 64, 96, 128]) torch.Size([16, 64, 96, 128])
#             # p2 torch.Size([16, 64, 48, 64]) torch.Size([16, 64, 48, 64])
#             # p3 torch.Size([16, 64, 24, 32]) torch.Size([16, 64, 24, 32])
#             # p4 torch.Size([16, 64, 12, 16]) torch.Size([16, 64, 12, 16])
#             # p5 torch.Size([16, 64, 6, 8]) torch.Size([16, 64, 6, 8])
#             # print(key, feats.shape, processed_feats.shape)

#         return feats_mat_seg_dict_processed


