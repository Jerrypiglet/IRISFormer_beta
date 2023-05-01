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
import models_def.models_light as models_light 
import models_def.model_matcls as model_matcls

from models_def.model_dpt.models import DPTBRDFModel, get_BRDFNet_DPT, get_LightNet_DPT

from icecream import ic
import torch.utils.checkpoint as cp

from models_def.model_dpt.blocks import forward_vit
import models_def.BilateralLayer as bs

class Model_Joint(nn.Module):
    def __init__(self, opt, logger):
        super(Model_Joint, self).__init__()
        self.opt = opt
        self.cfg = opt.cfg
        self.logger = logger
        self.non_learnable_layers = {}

        self.load_brdf_gt = self.opt.cfg.DATA.load_brdf_gt

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

        if self.cfg.MODEL_BRDF.enable:
            in_channels = 3
            if self.opt.cfg.MODEL_MATSEG.use_as_input:
                in_channels += 1
            if self.opt.cfg.MODEL_SEMSEG.use_as_input:
                in_channels += 1
            if self.opt.cfg.MODEL_MATSEG.use_pred_as_input:
                in_channels += 1

            self.encoder_to_use = models_brdf.encoder0
            self.decoder_to_use = models_brdf.decoder0

            if self.opt.cfg.MODEL_BRDF.DPT_baseline.enable:
                default_models = {
                    # "midas_v21": "dpt_weights/midas_v21-f6b98070.pt",
                    "dpt_base": self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_base_path,
                    "dpt_large": self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_large_path,
                    "dpt_hybrid": self.opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid_path,
                }
                
                model_type = self.opt.cfg.MODEL_BRDF.DPT_baseline.model
                model_path = str(Path(self.opt.cfg.PATH.pretrained_path) / default_models[model_type]) if default_models[model_type]!='NA' else None
                if_non_negative = True if self.opt.cfg.MODEL_BRDF.DPT_baseline.modality in ['de'] else False

                if model_type=='dpt_hybrid':
                    assert self.opt.cfg.MODEL_BRDF.DPT_baseline.modality == 'enabled', 'only support this mode for now; choose modes in MODEL_BRDF.enable_list'
                    self.BRDF_Net = get_BRDFNet_DPT(
                        opt=opt, 
                        model_path=model_path, 
                        modalities=self.opt.cfg.MODEL_BRDF.enable_list, 
                        backbone="vitb_rn50_384" if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.N_layers == -1 else "vitb_rn50_384_N_layers"
                    )
                elif model_type=='dpt_large':
                    # self.BRDF_Net = DPTBRDFModel(
                    #     opt=opt, 
                    #     cfg_DPT=opt.cfg.MODEL_BRDF.DPT_baseline, 
                    #     modality=self.opt.cfg.MODEL_BRDF.DPT_baseline.modality, 
                    #     path=model_path,
                    #     backbone="vitl16_384",
                    #     non_negative=if_non_negative,
                    #     enable_attention_hooks=False,
                    #     skip_keys=['scratch.output_conv'] if self.opt.cfg.MODEL_BRDF.DPT_baseline.if_skip_last_conv else [], 
                    # )
                    assert self.opt.cfg.MODEL_BRDF.DPT_baseline.modality == 'enabled', 'only support this mode for now; choose modes in MODEL_BRDF.enable_list'
                    self.BRDF_Net = get_BRDFNet_DPT(
                        opt=opt, 
                        model_path=model_path, 
                        modalities=self.opt.cfg.MODEL_BRDF.enable_list, 
                        backbone="vitl16_384"
                    )

                elif model_type=='dpt_base':
                    # self.BRDF_Net = DPTBRDFModel(
                    #     opt=opt, 
                    #     modality=self.opt.cfg.MODEL_BRDF.DPT_baseline.modality, 
                    #     path=model_path,
                    #     backbone="vitb16_384",
                    #     non_negative=if_non_negative,
                    #     enable_attention_hooks=False,
                    #     skip_keys=['scratch.output_conv'] if self.opt.cfg.MODEL_BRDF.DPT_baseline.if_skip_last_conv else [], 
                    # )
                    assert self.opt.cfg.MODEL_BRDF.DPT_baseline.modality == 'enabled', 'only support this mode for now; choose modes in MODEL_BRDF.enable_list'
                    self.BRDF_Net = get_BRDFNet_DPT(
                        opt=opt, 
                        model_path=model_path, 
                        modalities=self.opt.cfg.MODEL_BRDF.enable_list, 
                        backbone="vitb16_384"
                    )

                else:
                    print(model_type=='dpt_hybrid')
                    assert False, 'Unsupported model_type: %s!'%model_type

                # if dpt_optimize:
                #     self.BRDF_Net = self.BRDF_Net.to(memory_format=torch.channels_last)
                    # self.BRDF_Net = self.BRDF_Net.half()

                if self.cfg.MODEL_BRDF.DPT_baseline.if_freeze_backbone:
                    self.turn_off_names(['BRDF_Net.pretrained.model.patch_embed.backbone']) # patchembed backbone in DPT_hybrid (resnet)
                    freeze_bn_in_module(self.BRDF_Net.pretrained.model.patch_embed.backbone)

                if self.cfg.MODEL_BRDF.DPT_baseline.if_freeze_pretrained:
                    self.turn_off_names(['BRDF_Net.pretrained'])
                    freeze_bn_in_module(self.BRDF_Net.pretrained)

            else:
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
                        'encoder': self.encoder_to_use(opt, cascadeLevel = self.opt.cascadeLevel, in_channels = in_channels)
                        })

                if self.cfg.MODEL_BRDF.enable_BRDF_decoders:
                    if 'al' in self.cfg.MODEL_BRDF.enable_list:
                        self.BRDF_Net.update({'albedoDecoder': self.decoder_to_use(opt, mode=0, modality='al')})
                        if self.cfg.MODEL_BRDF.if_bilateral:
                            self.BRDF_Net.update({'albedoBs': bs.BilateralLayer(mode = 0)})

                    if 'no' in self.cfg.MODEL_BRDF.enable_list:
                        self.BRDF_Net.update({'normalDecoder': self.decoder_to_use(opt, mode=1, modality='no')})
                        if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                            self.BRDF_Net.update({'normalBs': bs.BilateralLayer(mode = 1)})

                    if 'ro' in self.cfg.MODEL_BRDF.enable_list:
                        self.BRDF_Net.update({'roughDecoder': self.decoder_to_use(opt, mode=2, modality='ro')})
                        if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                            self.BRDF_Net.update({'roughBs': bs.BilateralLayer(mode = 2)})

                    if 'de' in self.cfg.MODEL_BRDF.enable_list:
                        # if self.cfg.MODEL_BRDF.use_scale_aware_depth:
                        assert self.cfg.MODEL_BRDF.depth_activation in ['relu', 'sigmoid', 'tanh']
                        if self.cfg.MODEL_BRDF.depth_activation == 'relu':
                            self.BRDF_Net.update({'depthDecoder': self.decoder_to_use(opt, mode=5, modality='de')}) # default # -> [0, inf]
                        elif self.cfg.MODEL_BRDF.depth_activation == 'sigmoid':
                            self.BRDF_Net.update({'depthDecoder': self.decoder_to_use(opt, mode=6, modality='de')}) # -> [0, inf]
                        elif self.cfg.MODEL_BRDF.depth_activation == 'tanh':
                            self.BRDF_Net.update({'depthDecoder': self.decoder_to_use(opt, mode=4, modality='de')}) # -> [-1, 1]
                        if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                            self.BRDF_Net.update({'depthBs': bs.BilateralLayer(mode = 4)})
                        
                if self.cfg.MODEL_BRDF.enable_semseg_decoder:
                    self.BRDF_Net.update({'semsegDecoder': self.decoder_to_use(opt, mode=-1, out_channel=self.cfg.MODEL_SEMSEG.semseg_classes, if_PPM=self.cfg.MODEL_BRDF.semseg_PPM)})

            if self.cfg.MODEL_BRDF.if_freeze:
                self.BRDF_Net.eval()

        # self.guide_net = guideNet(opt)


        if self.cfg.MODEL_LIGHT.load_pretrained_MODEL_BRDF:
            self.load_pretrained_MODEL_BRDF(if_load_encoder=self.cfg.MODEL_BRDF.pretrained_if_load_encoder, if_load_decoder=self.cfg.MODEL_BRDF.pretrained_if_load_decoder, if_load_Bs=self.cfg.MODEL_BRDF.pretrained_if_load_Bs)

        if self.cfg.MODEL_LIGHT.enable:
            if self.cfg.MODEL_LIGHT.DPT_baseline.enable:
                assert opt.cascadeLevel == 0
                model_type = self.opt.cfg.MODEL_LIGHT.DPT_baseline.model
                if model_type=='dpt_hybrid':
                    self.LIGHT_Net = get_LightNet_DPT(
                        opt=opt, 
                        SGNum=opt.cfg.MODEL_LIGHT.SGNum, 
                        model_path=None, 
                        modalities=opt.cfg.MODEL_LIGHT.enable_list, 
                        backbone="vitb_rn50_384" if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.N_layers == -1 else "vitb_rn50_384_N_layers"
                    )
                elif model_type=='dpt_base':
                    self.LIGHT_Net = get_LightNet_DPT(
                        opt=opt, 
                        SGNum=opt.cfg.MODEL_LIGHT.SGNum, 
                        model_path=None, 
                        modalities=opt.cfg.MODEL_LIGHT.enable_list, 
                        backbone="vitb16_384", 
                    )
                elif model_type=='swin':
                    self.LIGHT_Net = get_LightNet_Swin(
                        opt=opt, 
                        SGNum=opt.cfg.MODEL_LIGHT.SGNum, 
                        modalities=opt.cfg.MODEL_LIGHT.enable_list, 
                    )
                else:
                    assert False, 'not supported yet!'
            else:
                self.LIGHT_Net = nn.ModuleDict({})
                self.LIGHT_Net.update({'lightEncoder':  models_light.encoderLight(cascadeLevel = opt.cascadeLevel, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
                if 'axis' in opt.cfg.MODEL_LIGHT.enable_list:
                    self.LIGHT_Net.update({'axisDecoder':  models_light.decoderLight(mode=0, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
                if 'lamb' in opt.cfg.MODEL_LIGHT.enable_list:
                    self.LIGHT_Net.update({'lambDecoder':  models_light.decoderLight(mode = 1, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})
                if 'weight' in opt.cfg.MODEL_LIGHT.enable_list:
                    self.LIGHT_Net.update({'weightDecoder':  models_light.decoderLight(mode = 2, SGNum = opt.cfg.MODEL_LIGHT.SGNum )})

            if self.cfg.MODEL_LIGHT.DPT_baseline.enable_as_single_est:
                self.LIGHT_WEIGHT_Net = get_LightNet_DPT(
                    opt=opt, 
                    SGNum=opt.cfg.MODEL_LIGHT.SGNum, 
                    model_path=None, 
                    modalities=[self.cfg.MODEL_LIGHT.DPT_baseline.enable_as_single_est_modality], 
                    backbone="vitb16_384", 
                )

            self.non_learnable_layers['renderLayer'] = models_light.renderingLayer(isCuda = opt.if_cuda, 
                imWidth=opt.cfg.MODEL_LIGHT.envCol, imHeight=opt.cfg.MODEL_LIGHT.envRow, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight)
            self.non_learnable_layers['output2env'] = models_light.output2env(isCuda = opt.if_cuda, 
                envWidth = opt.cfg.MODEL_LIGHT.envWidth, envHeight = opt.cfg.MODEL_LIGHT.envHeight, SGNum = opt.cfg.MODEL_LIGHT.SGNum )

            # if not self.opt.cfg.MODEL_LIGHT.use_GT_brdf:
            if self.cfg.MODEL_LIGHT.freeze_BRDF_Net:
                self.turn_off_names(['BRDF_Net'])
                freeze_bn_in_module(self.BRDF_Net)

            if self.cfg.MODEL_LIGHT.if_freeze or self.cfg.MODEL_LIGHT.DPT_baseline.enable_as_single_est_freeze_LightNet:
                self.turn_off_names(['LIGHT_Net'])
                freeze_bn_in_module(self.LIGHT_Net)

            if self.cfg.MODEL_LIGHT.load_pretrained_MODEL_LIGHT:
                self.load_pretrained_MODEL_LIGHT()
        if self.cfg.MODEL_MATCLS.enable:
            self.MATCLS_NET = model_matcls.netCS(opt=opt, inChannels=4, base_model=resnet.resnet34, if_est_scale=False, if_est_sup = opt.cfg.MODEL_MATCLS.if_est_sup)

    def freeze_BN(self):
        if self.cfg.MODEL_LIGHT.freeze_BRDF_Net:
            self.turn_off_names(['BRDF_Net'])
            freeze_bn_in_module(self.BRDF_Net)

        if self.cfg.MODEL_LIGHT.if_freeze or self.cfg.MODEL_LIGHT.DPT_baseline.enable_as_single_est_freeze_LightNet:
            self.turn_off_names(['LIGHT_Net'])
            freeze_bn_in_module(self.LIGHT_Net)


    def forward(self, input_dict, if_has_gt_BRDF=True):
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

        if self.cfg.MODEL_BRDF.enable:
            if self.cfg.MODEL_BRDF.if_freeze:
                self.BRDF_Net.eval()
            input_dict_extra = {'input_dict_guide': input_dict_guide}

            if self.cfg.MODEL_BRDF.DPT_baseline.enable:
                return_dict_brdf = self.forward_brdf_DPT_baseline(input_dict, input_dict_extra=input_dict_extra, if_has_gt_BRDF=if_has_gt_BRDF)
            else:
                return_dict_brdf = self.forward_brdf(input_dict, input_dict_extra=input_dict_extra, if_has_gt_BRDF=if_has_gt_BRDF)
        else:
            return_dict_brdf = {}
        return_dict.update(return_dict_brdf)

        if self.cfg.MODEL_LIGHT.enable:
            if self.cfg.MODEL_LIGHT.if_freeze or self.cfg.MODEL_LIGHT.DPT_baseline.enable_as_single_est_freeze_LightNet:
                self.LIGHT_Net.eval()
            if self.opt.cfg.DATASET.if_no_gt_BRDF:
                return_dict_light = self.forward_light_real(input_dict, return_dict_brdf=return_dict_brdf)
            else:
                return_dict_light = self.forward_light(input_dict, return_dict_brdf=return_dict_brdf)
        else:
            return_dict_light = {}
        return_dict.update(return_dict_light)
        
        if self.cfg.MODEL_MATCLS.enable:
            return_dict_matcls = self.forward_matcls(input_dict)
            return_dict.update(return_dict_matcls)

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

    def forward_brdf_swin(self, input_dict, input_dict_extra={}):
        img_batch = input_dict['imBatch']
        # backbone_output_tuple = self.BRDF_Net.backbone(img_batch)
        # backbone_output_list = list(backbone_output_tuple)
        # # 1/4 torch.Size([2, 96, 64, 80])
        # # 1/8 torch.Size([2, 192, 32, 40])
        # # 1/16 torch.Size([2, 384, 16, 20])
        # # 1/32 torch.Size([2, 768, 8, 10])
        # decoder_output = self.BRDF_Net.decoder(backbone_output_tuple)
        # # print(decoder_output.shape) # torch.Size([2, 3, 64, 80])

        # for a in backbone_output_list:
        #     print(a.shape)

        return_dict = self.BRDF_Net(img_batch)
        return return_dict

    def forward_brdf_DPT_baseline(self, input_dict, input_dict_extra={}, if_has_gt_BRDF=True):
        return_dict = {}
        img_batch = input_dict['imBatch']
        input_dict_extra.update({'brdf_loss_mask': input_dict['brdf_loss_mask'], 'input_dict': input_dict})
        # print(img_batch.shape)
        # img_batch = input_dict['imBatch'].half()
        # img_input = dpt_transform({"image": img_batch})["image"]
        # dpt_prediction, extra_DPT_return_dict = self.BRDF_Net.forward(img_batch, input_dict_extra=input_dict_extra)
        # return_dict = self.BRDF_Net.forward(img_batch, input_dict_extra=input_dict_extra)

        return_dicts = {}
        modalities = self.opt.cfg.MODEL_BRDF.enable_list
        if self.opt.cfg.MODEL_BRDF.DPT_baseline.model == 'swin':
            assert modalities == ['al']
            return_dicts['al'] = self.forward_brdf_swin(input_dict, input_dict_extra={})
        else:
            if self.cfg.MODEL_BRDF.DPT_baseline.modality=='enabled':
                if self.opt.cfg.MODEL_BRDF.DPT_baseline.if_share_pretrained:
                    module_hooks_dict = {}
                    input_dict_extra['shared_pretrained'] = forward_vit(self.opt, self.opt.cfg.MODEL_BRDF.DPT_baseline, self.BRDF_Net.shared_pretrained, img_batch, input_dict_extra={**input_dict_extra, **module_hooks_dict})
                elif self.cfg.MODEL_BRDF.DPT_baseline.if_share_patchembed:
                    x = self.BRDF_Net.shared_patch_embed_backbone(img_batch)
                    input_dict_extra['shared_patch_embed_backbone_output'] = x
                
                for modality in modalities:
                    return_dicts[modality] = self.BRDF_Net[modality].forward(img_batch, input_dict_extra=input_dict_extra)
            else:
                assert False
                # modality = self.cfg.MODEL_BRDF.DPT_baseline.modality
                # modalities = [modality]
                # return_dicts = {modality: self.BRDF_Net.forward(img_batch, input_dict_extra=input_dict_extra)}

        for modality in modalities:
            dpt_prediction, extra_DPT_return_dict = return_dicts[modality]
            if modality == 'al':
                albedoPred = 0.5 * (dpt_prediction + 1)
                # if (not self.opt.cfg.DATASET.if_no_gt_semantics):
                # print(input_dict['segBRDFBatch'].shape, input_dict['albedoBatch'].shape)
                albedoPred = torch.clamp(albedoPred, 0, 1)
                return_dict.update({'albedoPred': albedoPred})
                # if not self.cfg.MODEL_BRDF.use_scale_aware_albedo:
                # print(input_dict['segBRDFBatch'].shape, albedoPred.shape)
                if not self.opt.cfg.DATASET.if_no_gt_BRDF:
                    input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
                    albedoPred_aligned = models_brdf.LSregress(albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred),
                            input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoPred)
                    albedoPred_aligned = torch.clamp(albedoPred_aligned, 0, 1)
                    return_dict.update({'albedoPred_aligned': albedoPred_aligned})
            elif modality == 'de':
                # if self.cfg.MODEL_BRDF.use_scale_aware_depth:
                depthPred = dpt_prediction
                return_dict.update({'depthPred': depthPred})
                # else:
                # depthPred = 0.5 * (dpt_prediction + 1) # [-1, 1] -> [0, 1]
                if not self.opt.cfg.DATASET.if_no_gt_BRDF:
                    depthPred_aligned = models_brdf.LSregress(depthPred *  input_dict['segAllBatch'].expand_as(depthPred),
                            input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthPred)
                    return_dict.update({'depthPred_aligned': depthPred_aligned})
            elif modality == 'ro':
                roughPred = dpt_prediction
                return_dict.update({'roughPred': roughPred})
            elif modality == 'no':
                normalPred = dpt_prediction
                return_dict.update({'normalPred': normalPred})
            else:
                assert False, 'Unsupported modality: %s'%modality

        # print(extra_DPT_return_dict.keys())
        return_dict.update({'albedo_extra_output_dict': {}})
        if 'matseg_affinity' in extra_DPT_return_dict:
            return_dict['albedo_extra_output_dict'].update({'matseg_affinity': extra_DPT_return_dict['matseg_affinity']})

        return return_dict

    def forward_brdf(self, input_dict, input_dict_extra={}, if_has_gt_BRDF=True):
        if_has_gt_BRDF = if_has_gt_BRDF and (not self.opt.cfg.DATASET.if_no_gt_BRDF) and self.load_brdf_gt and not self.opt.cfg.DEBUG.if_test_real
        if_has_gt_segBRDF = if_has_gt_BRDF and not self.opt.cfg.DEBUG.if_nyud and not self.opt.cfg.DEBUG.if_iiw and not self.opt.cfg.DEBUG.if_test_real

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
            
        x1, x2, x3, x4, x5, x6, extra_output_dict = self.BRDF_Net['encoder'](input_tensor, input_dict_extra=input_dict_extra)

        return_dict = {'encoder_outputs': {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'brdf_extra_output_dict': extra_output_dict}}
        albedo_output = {}

        if self.cfg.MODEL_BRDF.enable_BRDF_decoders:
            # input_dict_extra = {}
            if input_dict_guide is not None:
                input_dict_extra.update({'input_dict_guide': input_dict_guide})

            # print(input_dict['segBRDFBatch'].shape, input_dict['segAllBatch'].shape)
            if 'al' in self.cfg.MODEL_BRDF.enable_list:
                albedo_output = self.BRDF_Net['albedoDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_extra=input_dict_extra)
                albedoPred = 0.5 * (albedo_output['x_out'] + 1)

                if if_has_gt_segBRDF:
                    input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
                
                albedoPred = torch.clamp(albedoPred, 0, 1)
                return_dict.update({'albedoPred': albedoPred})
                # if not self.cfg.MODEL_BRDF.use_scale_aware_albedo:
                if if_has_gt_BRDF and if_has_gt_segBRDF:
                    albedoPred_aligned = models_brdf.LSregress(albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred),
                            input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoPred)
                    albedoPred_aligned = torch.clamp(albedoPred_aligned, 0, 1)
                    return_dict.update({'albedoPred_aligned': albedoPred_aligned, 'albedo_extra_output_dict': albedo_output['extra_output_dict']})

                if self.cfg.MODEL_BRDF.if_bilateral:
                    albedoBsPred, albedoConf = self.BRDF_Net['albedoBs'](input_dict['imBatch'], albedoPred.detach(), albedoPred )
                    if if_has_gt_BRDF and 'al' in self.opt.cfg.DATA.data_read_list:
                        albedoBsPred = models_brdf.LSregress(albedoBsPred * input_dict['segBRDFBatch'].expand_as(albedoBsPred ),
                            input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoBsPred )
                    albedoBsPred = torch.clamp(albedoBsPred, 0, 1 )
                    if if_has_gt_segBRDF:
                        albedoBsPred = input_dict['segBRDFBatch'] * albedoBsPred
                    return_dict.update({'albedoBsPred': albedoBsPred, 'albedoConf': albedoConf})

                    if if_has_gt_BRDF and 'al' in self.opt.cfg.DATA.data_read_list:
                        albedoBsPred_aligned, albedoConf_aligned = self.BRDF_Net['albedoBs'](input_dict['imBatch'], albedoPred_aligned.detach(), albedoPred_aligned )
                        return_dict.update({'albedoBsPred_aligned': albedoBsPred_aligned})

            if 'no' in self.cfg.MODEL_BRDF.enable_list:
                normal_output = self.BRDF_Net['normalDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_extra=input_dict_extra)
                normalPred = normal_output['x_out']
                return_dict.update({'normalPred': normalPred, 'normal_extra_output_dict': normal_output['extra_output_dict']})
                if if_has_gt_BRDF:
                    if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                        normalBsPred = normalPred.clone().detach()
                        normalConf = albedoConf.clone().detach()
                        return_dict.update({'normalBsPred': normalBsPred, 'normalConf': normalConf})

            if 'ro' in self.cfg.MODEL_BRDF.enable_list:
                rough_output = self.BRDF_Net['roughDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_extra=input_dict_extra)
                roughPred = rough_output['x_out']
                return_dict.update({'roughPred': roughPred, 'rough_extra_output_dict': rough_output['extra_output_dict']})
                # if if_has_gt_BRDF:
                if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                    roughBsPred, roughConf = self.BRDF_Net['roughBs'](input_dict['imBatch'], albedoPred.detach(), 0.5*(roughPred+1.) )
                    roughBsPred = torch.clamp(2 * roughBsPred - 1, -1, 1)
                    return_dict.update({'roughBsPred': roughBsPred, 'roughConf': roughConf})

            if 'de' in self.cfg.MODEL_BRDF.enable_list:
                depth_output = self.BRDF_Net['depthDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_extra=input_dict_extra)
                depthPred = depth_output['x_out']
                # if not self.cfg.MODEL_BRDF.use_scale_aware_depth:
                return_dict.update({'depthPred': depthPred})

                if if_has_gt_BRDF:
                    if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                        depthBsPred, depthConf = self.BRDF_Net['depthBs'](input_dict['imBatch'], albedoPred.detach(), 0.5*(depthPred+1.) if self.cfg.MODEL_BRDF.depth_activation=='tanh' else depthPred )
                        if if_has_gt_segBRDF:
                            depthBsPred = models_brdf.LSregress(depthBsPred *  input_dict['segAllBatch'].expand_as(depthBsPred),
                                    input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthBsPred)
                        else:
                            depthBsPred = models_brdf.LSregress(depthBsPred, input_dict['depthBatch'], depthBsPred)
                        return_dict.update({'depthBsPred': depthBsPred, 'depthConf': depthConf})
                    
                    if self.cfg.MODEL_BRDF.depth_activation == 'tanh':
                        depthPred_aligned = 0.5 * (depthPred + 1) # [-1, 1] -> [0, 1]
                        # print(torch.max(depthPred_aligned), torch.min(depthPred_aligned), torch.median(depthPred_aligned))
                    else:
                        depthPred_aligned = depthPred # [0, inf]
                    if if_has_gt_segBRDF:
                        depthPred_aligned = models_brdf.LSregress(depthPred_aligned *  input_dict['segAllBatch'].expand_as(depthPred_aligned),
                                input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthPred_aligned)
                    else:
                        depthPred_aligned = models_brdf.LSregress(depthPred_aligned, input_dict['depthBatch'], depthPred_aligned)
                    # print('-->', torch.max(depthPred_aligned), torch.min(depthPred_aligned), torch.median(depthPred_aligned))
                    return_dict.update({'depthPred_aligned': depthPred_aligned, 'depth_extra_output_dict': depth_output['extra_output_dict']})

                if self.cfg.MODEL_BRDF.if_bilateral and not self.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                    # assert self.cfg.DEBUG.if_test_real
                    if 'albedoPred' in return_dict:
                        depthBsPred, depthConf = self.BRDF_Net['depthBs'](input_dict['imBatch'], return_dict['albedoPred'].detach(), depthPred )
                    else:
                        assert self.opt.cfg.DEBUG.if_load_dump_BRDF_offline
                        depthBsPred, depthConf = self.BRDF_Net['depthBs'](input_dict['imBatch'], input_dict['albedoBatch'].detach(), depthPred )
                    return_dict.update({'depthBsPred': depthBsPred})
                    if if_has_gt_BRDF and 'de' in self.opt.cfg.DATA.data_read_list:
                        assert 'depthPred_aligned' in return_dict
                        depthBsPred_aligned, depthConf = self.BRDF_Net['depthBs'](input_dict['imBatch'], return_dict['albedoPred'].detach(), depthPred_aligned )
                        return_dict.update({'depthBsPred_aligned': depthBsPred_aligned})



            # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape)
            # return_dict.update({'albedoPred': albedosPred, 'normalPred': normalPred, 'roughPred': roughPred, 'depthPred': depthPred})

        if self.cfg.MODEL_BRDF.enable_semseg_decoder:
            semsegPred = self.BRDF_Net['semsegDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6)['x_out']
            return_dict.update({'semseg_pred': semsegPred})
            
        return return_dict

    def forward_LIGHT_Net(self, input_dict, imBatch, albedoInput, depthInput, normalInput, roughInput, ):
        # print(imBatch.shape, albedoInput.shape, depthInput.shape, normalInput.shape, roughInput.shape)
        # if self.cfg.DATA.if_pad_to_32x:
        #     imBatch = imBatch[:, :, :self.cfg.DATA.im_height, :self.cfg.DATA.im_width].contiguous()
        #     albedoInput = albedoInput[:, :, :self.cfg.DATA.im_height, :self.cfg.DATA.im_width].contiguous()
        #     depthInput = depthInput[:, :, :self.cfg.DATA.im_height, :self.cfg.DATA.im_width].contiguous()
        #     normalInput = normalInput[:, :, :self.cfg.DATA.im_height, :self.cfg.DATA.im_width].contiguous()
        #     roughInput = roughInput[:, :, :self.cfg.DATA.im_height, :self.cfg.DATA.im_width].contiguous()
            # print('-->', imBatch.shape, albedoInput.shape, depthInput.shape, normalInput.shape, roughInput.shape)
            
        # note: normalization/rescaling also needed for GT BRDFs
        # bn, ch, nrow, ncol = albedoInput.size()
        # albedoInput = albedoInput.view(bn, -1)
        # albedoInput = albedoInput / torch.clamp(torch.mean(albedoInput, dim=1), min=1e-10).unsqueeze(1) / 3.0
        # albedoInput = albedoInput.view(bn, ch, nrow, ncol)

        # bn, ch, nrow, ncol = depthInput.size()
        # depthInput = depthInput.view(bn, -1)
        # depthInput = depthInput / torch.clamp(torch.mean(depthInput, dim=1), min=1e-10).unsqueeze(1) / 3.0
        # depthInput = depthInput.view(bn, ch, nrow, ncol)
        
        # print(albedoInput.shape, torch.max(albedoInput), torch.min(albedoInput), torch.median(albedoInput))
        # print(depthInput.shape, torch.max(depthInput), torch.min(depthInput), torch.median(depthInput))
        # print(normalInput.shape, torch.max(normalInput), torch.min(normalInput), torch.median(normalInput))
        # print(roughInput.shape, torch.max(roughInput), torch.min(roughInput), torch.median(roughInput))

        im_h, im_w = self.cfg.DATA.im_height, self.cfg.DATA.im_width
        assert self.cfg.DATA.pad_option == 'const'

        # bn, ch, nrow, ncol = albedoInput.size()
        # if not self.cfg.MODEL_LIGHT.use_scale_aware_loss:
        albedoInput[:, :, :im_h, :im_w] = albedoInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(albedoInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0

        # bn, ch, nrow, ncol = depthInput.size()
        depthInput[:, :, :im_h, :im_w] = depthInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(depthInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0

        normalInput[:, :, :im_h, :im_w] =  0.5 * (normalInput[:, :, :im_h, :im_w] + 1)
        roughInput[:, :, :im_h, :im_w] =  0.5 * (roughInput[:, :, :im_h, :im_w] + 1)

        imBatchLarge = F.interpolate(imBatch, scale_factor=2, mode='bilinear')
        albedoInputLarge = F.interpolate(albedoInput, scale_factor=2, mode='bilinear')
        depthInputLarge = F.interpolate(depthInput, scale_factor=2, mode='bilinear')
        normalInputLarge = F.interpolate(normalInput, scale_factor=2, mode='bilinear')
        roughInputLarge = F.interpolate(roughInput, scale_factor=2, mode='bilinear')

        input_batch = torch.cat([imBatchLarge, albedoInputLarge, normalInputLarge, roughInputLarge, depthInputLarge ], dim=1 )

        if self.opt.cascadeLevel == 0:
            # print(input_batch.shape)
            x1, x2, x3, x4, x5, x6 = self.LIGHT_Net['lightEncoder'](input_batch.detach() )
        else:
            assert self.opt.cascadeLevel > 0
            x1, x2, x3, x4, x5, x6 = self.LIGHT_Net['lightEncoder'](input_batch.detach(), input_dict['envmapsPreBatch'].detach() )

        # print(input_batch.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape) # torch.Size([4, 11, 480, 640]) torch.Size([4, 128, 60, 80]) torch.Size([4, 256, 30, 40]) torch.Size([4, 256, 15, 20]) torch.Size([4, 512, 7, 10]) torch.Size([4, 512, 3, 5]) torch.Size([4, 1024, 3, 5])

        # Prediction
        if 'axis' in self.cfg.MODEL_LIGHT.enable_list and not self.cfg.MODEL_LIGHT.use_GT_light_sg:
            axisPred_ori = self.LIGHT_Net['axisDecoder'](x1, x2, x3, x4, x5, x6) # torch.Size([4, 12, 3, 120, 160])
        else:
            axisPred_ori = input_dict['sg_axis_Batch'] # (4, 120, 160, 12, 3)
            axisPred_ori = axisPred_ori.permute(0, 3, 4, 1, 2)
        if 'lamb' in self.cfg.MODEL_LIGHT.enable_list and not self.cfg.MODEL_LIGHT.use_GT_light_sg:
            lambPred_ori = self.LIGHT_Net['lambDecoder'](x1, x2, x3, x4, x5, x6) # torch.Size([4, 12, 120, 160])
        else:
            lambPred_ori = input_dict['sg_lamb_Batch'] # (4, 120, 160, 12, 1)
            lambPred_ori = lambPred_ori.squeeze(4).permute(0, 3, 1, 2)

        if 'weight' in self.cfg.MODEL_LIGHT.enable_list and not self.cfg.MODEL_LIGHT.use_GT_light_sg:
            weightPred_ori = self.LIGHT_Net['weightDecoder'](x1, x2, x3, x4, x5, x6) # torch.Size([4, 36, 120, 160])
        else:
            weightPred_ori = input_dict['sg_weight_Batch'] # (4, 120, 160, 12, 3)
            weightPred_ori = weightPred_ori.flatten(3).permute(0, 3, 1, 2)
            # weightPred_ori = torch.ones_like(weightPred_ori).cuda() * 0.1
            # weightPred_ori[weightPred_ori>500] = 500.
            # weightPred_ori = weightPred_ori / 500.
        # print(torch.max(weightPred_ori), torch.min(weightPred_ori), torch.median(weightPred_ori))

        # print(axisPred_ori.shape, lambPred_ori.shape, weightPred_ori.shape)
        return axisPred_ori, lambPred_ori, weightPred_ori


    def forward_LIGHT_Net_DPT_baseline(self, input_dict, imBatch, albedoInput, depthInput, normalInput, roughInput, ):
        # if self.cfg.DATA.if_pad_to_32x:
        #     imBatch = imBatch[:, :, :im_h, :im_w].contiguous()
        #     albedoInput = albedoInput[:, :, :im_h, :im_w].contiguous()
        #     depthInput = depthInput[:, :, :im_h, :im_w].contiguous()
        #     normalInput = normalInput[:, :, :im_h, :im_w].contiguous()
        #     roughInput = roughInput[:, :, :im_h, :im_w].contiguous()
            # print('-->', imBatch.shape, albedoInput.shape, depthInput.shape, normalInput.shape, roughInput.shape)
            
        # note: normalization/rescaling also needed for GT BRDFs
        im_h, im_w = self.cfg.DATA.im_height, self.cfg.DATA.im_width
        assert self.cfg.DATA.pad_option == 'const'

        # bn, ch, nrow, ncol = albedoInput.size()
        # if not self.cfg.MODEL_LIGHT.use_scale_aware_loss:
        albedoInput[:, :, :im_h, :im_w] = albedoInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(albedoInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0
        # albedoInput = albedoInput / torch.clamp(
        #         torch.mean(albedoInput.flatten(1), dim=1, keepdim=True)
        #     , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0

        # bn, ch, nrow, ncol = depthInput.size()
        depthInput[:, :, :im_h, :im_w] = depthInput[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(depthInput[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0
        # depthInput = depthInput / torch.clamp(
        #         torch.mean(depthInput.flatten(1), dim=1, keepdim=True)
        #     , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0

        # imBatchLarge = F.interpolate(imBatch, [self.cfg.DATA.im_height*2, self.cfg.DATA.im_width*2], mode='bilinear')
        # albedoInputLarge = F.interpolate(albedoInput, [self.cfg.DATA.im_height*2, self.cfg.DATA.im_width*2], mode='bilinear')
        # depthInputLarge = F.interpolate(depthInput, [self.cfg.DATA.im_height*2, self.cfg.DATA.im_width*2], mode='bilinear')
        # normalInputLarge = F.interpolate(normalInput, [self.cfg.DATA.im_height*2, self.cfg.DATA.im_width*2], mode='bilinear')
        # roughInputLarge = F.interpolate(roughInput, [self.cfg.DATA.im_height*2, self.cfg.DATA.im_width*2], mode='bilinear')

        normalInput[:, :, :im_h, :im_w] =  0.5 * (normalInput[:, :, :im_h, :im_w] + 1)
        roughInput[:, :, :im_h, :im_w] =  0.5 * (roughInput[:, :, :im_h, :im_w] + 1)
        
        input_batch = torch.cat([imBatch, albedoInput, normalInput, roughInput, depthInput ], dim=1 ).detach()
        # input_batch = imBatch

        input_dict_extra = {}
        # input_dict_extra.update({'input_dict': input_dict})

        modalities = self.opt.cfg.MODEL_LIGHT.enable_list
        return_dicts = {}

        if self.opt.cfg.MODEL_LIGHT.DPT_baseline.model == 'swin':
            if self.opt.cfg.MODEL_LIGHT.DPT_baseline.if_share_pretrained:
                module_hooks_dict = {}
                input_dict_extra['shared_pretrained'] = self.LIGHT_Net.shared_pretrained(input_batch)
            for modality in modalities:
                return_dicts[modality] = self.LIGHT_Net[modality].forward(input_batch, input_dict_extra=input_dict_extra)
        else:
            if self.opt.cfg.MODEL_LIGHT.DPT_baseline.if_share_pretrained:
                module_hooks_dict = {}
                # if self.opt.cfg.MODEL_LIGHT.DPT_baseline.if_checkpoint:
                #     input_dict_extra['shared_pretrained'] = cp.checkpoint(forward_vit, self.opt, self.opt.cfg.MODEL_LIGHT.DPT_baseline, self.LIGHT_Net.shared_pretrained, input_batch, {**input_dict_extra, **module_hooks_dict})
                # else:
                input_dict_extra['shared_pretrained'] = forward_vit(self.opt, self.opt.cfg.MODEL_LIGHT.DPT_baseline, self.LIGHT_Net.shared_pretrained, input_batch, input_dict_extra={**input_dict_extra, **module_hooks_dict})
            elif self.cfg.MODEL_LIGHT.DPT_baseline.if_share_patchembed:
                # if self.opt.cfg.MODEL_LIGHT.DPT_baseline.if_checkpoint:
                #     x = cp.checkpoint(self.LIGHT_Net.shared_patch_embed_backbone, input_batch)
                # else:
                x = self.LIGHT_Net.shared_patch_embed_backbone(input_batch)
                input_dict_extra['shared_patch_embed_backbone_output'] = x
            
            for modality in modalities:
                # if self.opt.cfg.MODEL_LIGHT.DPT_baseline.if_checkpoint:
                #     return_dicts[modality] = cp.checkpoint(self.LIGHT_Net[modality].forward, input_batch, input_dict_extra)
                # else:
                return_dicts[modality] = self.LIGHT_Net[modality].forward(input_batch, input_dict_extra=input_dict_extra)

        assert self.opt.cascadeLevel == 0

        return return_dicts['axis'], return_dicts['lamb'], return_dicts['weight']

    def forward_LIGHT_SINGLE_Net_DPT_baseline(self, input_dict, imBatch, albedoPred, depthPred, normalPred, roughPred, ):
        assert False, 'disabled for now'
        im_h, im_w = self.cfg.DATA.im_height, self.cfg.DATA.im_width
        assert self.cfg.DATA.pad_option == 'const'

        albedoPred[:, :, :im_h, :im_w] = albedoPred[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(albedoPred[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0
        depthPred[:, :, :im_h, :im_w] = depthPred[:, :, :im_h, :im_w] / torch.clamp(
                torch.mean(depthPred[:, :, :im_h, :im_w].flatten(1), dim=1, keepdim=True)
            , min=1e-10).unsqueeze(-1).unsqueeze(-1) / 3.0
        normalPred[:, :, :im_h, :im_w] =  0.5 * (normalPred[:, :, :im_h, :im_w] + 1)
        roughPred[:, :, :im_h, :im_w] =  0.5 * (roughPred[:, :, :im_h, :im_w] + 1)
        
        input_batch = torch.cat([imBatch, albedoPred, normalPred, roughPred, depthPred ], dim=1 ).detach()

        input_dict_extra = {}
        input_dict_extra.update({'input_dict': input_dict})

        modality = self.cfg.MODEL_LIGHT.DPT_baseline.enable_as_single_est_modality
        return_dicts = {}

        return_dicts[modality] = self.LIGHT_WEIGHT_Net[modality].forward(input_batch, input_dict_extra=input_dict_extra)

        assert self.opt.cascadeLevel == 0

        return return_dicts[modality]

    def forward_light(self, input_dict, return_dict_brdf):
        im_h, im_w = self.cfg.DATA.im_height, self.cfg.DATA.im_width

        # Normalize Albedo and depth
        if 'al' in self.cfg.MODEL_BRDF.enable_list and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            albedoInput = return_dict_brdf['albedoPred'].detach().clone()
        else:
            albedoInput = input_dict['albedoBatch'].detach().clone()

        if 'de' in self.cfg.MODEL_BRDF.enable_list and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            depthInput = return_dict_brdf['depthPred'].detach().clone()
            # print('-', depthInput.shape, torch.max(depthInput), torch.min(depthInput), torch.median(depthInput))
            if self.cfg.MODEL_BRDF.depth_activation == 'tanh':
                depthInput = 0.5 * (depthInput + 1) # [-1, 1] -> [0, 1]
            # print('->', depthInput.shape, torch.max(depthInput), torch.min(depthInput), torch.median(depthInput))
        else:
            depthInput = input_dict['depthBatch'].detach().clone()

        if 'no' in self.cfg.MODEL_BRDF.enable_list and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            normalInput = return_dict_brdf['normalPred'].detach().clone()
        else:
            normalInput = input_dict['normalBatch'].detach().clone()

        if 'ro' in self.cfg.MODEL_BRDF.enable_list and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            roughInput = return_dict_brdf['roughPred'].detach().clone()
        else:
            roughInput = input_dict['roughBatch'].detach().clone()

        imBatch = input_dict['imBatch']
        segBRDFBatch = input_dict['segBRDFBatch']

        if self.cfg.MODEL_LIGHT.freeze_BRDF_Net and not self.cfg.MODEL_LIGHT.use_GT_brdf:
            assert self.BRDF_Net.training == False



        if self.cfg.MODEL_LIGHT.DPT_baseline.enable:
            axisPred_ori, lambPred_ori, weightPred_ori = self.forward_LIGHT_Net_DPT_baseline(input_dict, imBatch, albedoInput, depthInput, normalInput, roughInput)
        else:
            axisPred_ori, lambPred_ori, weightPred_ori = self.forward_LIGHT_Net(input_dict, imBatch, albedoInput, depthInput, normalInput, roughInput)
            # print(axisPred_ori.shape, lambPred_ori.shape, weightPred_ori.shape)

        # if_print = self.cfg.MODEL_LIGHT.DPT_baseline.enable
        if_print = False

        if self.opt.is_master and if_print:
            print('--(unet) weight', torch.max(weightPred_ori), torch.min(weightPred_ori), torch.median(weightPred_ori), weightPred_ori.shape)
            print('--(unet) lamb', torch.max(lambPred_ori), torch.min(lambPred_ori), torch.median(lambPred_ori), lambPred_ori.shape)
            print('--(unet) axis', torch.max(axisPred_ori), torch.min(axisPred_ori), torch.median(axisPred_ori), axisPred_ori.shape)

        if self.cfg.MODEL_LIGHT.DPT_baseline.enable_as_single_est:
            singlePred_ori = self.forward_LIGHT_SINGLE_Net_DPT_baseline(input_dict, imBatch, albedoInput, depthInput, normalInput, roughInput)
            modality = self.cfg.MODEL_LIGHT.DPT_baseline.enable_as_single_est_modality
            if modality=='weight':
                weightPred_ori = singlePred_ori
                # weightPred_ori = weightPred_ori / 10.
            elif modality=='axis':
                axisPred_ori = singlePred_ori
            elif modality=='lamb':
                lambPred_ori = singlePred_ori
            else:
                assert False

            # print(weightPred_ori.shape)

        if self.opt.is_master and if_print:
            print('--weight', torch.max(weightPred_ori), torch.min(weightPred_ori), torch.median(weightPred_ori), weightPred_ori.shape)
            # UNet: tensor(0.5139, device='cuda:0') tensor(0., device='cuda:0') tensor(0.0210, device='cuda:0') # CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 5324 --nproc_per_node=2 train/trainEmitter-20210928.py --eval_every_iter 500 --if_overfit_train True --if_val False --if_vis True --if_train False --task_name tmp --resume 20211004-153946--DATE-train_mm1_LightNet-gtBRDF-32x_ORminiOverfit-HDR_bs8on2_32workers DATASET.num_workers 16 SOLVER.ims_per_batch 2 TEST.ims_per_batch 2 MODEL_BRDF.enable True MODEL_BRDF.load_pretrained_pth False MODEL_BRDF.enable_BRDF_decoders True MODEL_BRDF.enable_list al_no_de_ro DATA.data_read_list al_no_de_ro DATA.im_height 240 DATA.im_width 320 train_h 240 train_w 320 SOLVER.lr 0.00005 MODEL_LIGHT.enable True MODEL_LIGHT.load_pretrained_MODEL_LIGHT False MODEL_LIGHT.use_GT_brdf True DATA.if_load_png_not_hdr False DATA.if_pad_to_32x True DATA.pad_option reflect DATASET.mini True
            # DPT-base: tensor(0.7339, device='cuda:1') tensor(0.2201, device='cuda:1') tensor(0.5054, device='cuda:1') # CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --master_port 5324 --nproc_per_node=2 train/trainEmitter-20210928.py --eval_every_iter 500 --if_overfit_train True --if_val False --if_vis True --if_train False --task_name tmp DATASET.num_workers 16 SOLVER.ims_per_batch 2 TEST.ims_per_batch 2 MODEL_BRDF.enable True MODEL_BRDF.load_pretrained_pth False MODEL_BRDF.enable_BRDF_decoders True MODEL_BRDF.enable_list al_no_de_ro DATA.data_read_list al_no_de_ro DATA.im_height 240 DATA.im_width 320 train_h 240 train_w 320 SOLVER.lr 0.00005 MODEL_LIGHT.enable True MODEL_LIGHT.load_pretrained_MODEL_LIGHT False MODEL_LIGHT.use_GT_brdf True DATA.if_load_png_not_hdr False DATA.if_pad_to_32x True DATA.pad_option reflect DATASET.mini True MODEL_LIGHT.DPT_baseline.enable True MODEL_LIGHT.DPT_baseline.if_share_pretrained True
            print('--axis', torch.max(axisPred_ori), torch.min(axisPred_ori), torch.median(axisPred_ori), axisPred_ori.shape)
            # UNet: tensor(1.0000, device='cuda:1') tensor(-1.0000, device='cuda:1') tensor(-0.0546, device='cuda:1')
            # DPT-base: tensor(1.0000, device='cuda:0') tensor(-1.0000, device='cuda:0') tensor(0.1689, device='cuda:0')
            print('--lamb', torch.max(lambPred_ori), torch.min(lambPred_ori), torch.median(lambPred_ori), lambPred_ori.shape)
            # UNet: tensor(1., device='cuda:0') tensor(0.1491, device='cuda:0') tensor(0.9809, device='cuda:0')
            # DPT-base: tensor(0.8526, device='cuda:0') tensor(0.3087, device='cuda:0') tensor(0.5454, device='cuda:0')

        # print(input_dict.keys())
        # print(input_dict['envmapsBatch'].shape, axisPred_ori.shape, lambPred_ori.shape, weightPred_ori.shape) # torch.Size([4, 3, 120, 160, 8, 16]) torch.Size([4, 12, 3, 120, 160]) torch.Size([4, 12, 120, 160]) torch.Size([4, 36, 120, 160])
        # print(axisPred_ori[0, 0, :, :2, :2])
        # print(lambPred_ori[0, :3, :2, :2])
        # print(weightPred_ori[0, :3, :2, :2])

        # bn, SGNum, _, envRow, envCol = axisPred_ori.size()
        # envmapsPred = torch.cat([axisPred.view(bn, SGNum * 3, envRow, envCol ), lambPred, weightPred], dim=1)

        # weightPred_ori = weightPred_ori / 50.
        
        if self.cfg.DATA.if_pad_to_32x:
            axisPred_ori, lambPred_ori, weightPred_ori = axisPred_ori[:, :, :, :im_h//2, :im_w//2], lambPred_ori[:, :, :im_h//2, :im_w//2], weightPred_ori[:, :, :im_h//2, :im_w//2]
            imBatch = imBatch[:, :, :im_h, :im_w]
            segBRDFBatch = segBRDFBatch[:, :, :im_h, :im_w]

        imBatchSmall = F.adaptive_avg_pool2d(imBatch, (self.cfg.MODEL_LIGHT.envRow, self.cfg.MODEL_LIGHT.envCol) )
        # segBRDFBatchSmall = F.adaptive_avg_pool2d(segBRDFBatch, (self.cfg.MODEL_LIGHT.envRow, self.cfg.MODEL_LIGHT.envCol) )
        segBRDFBatchSmall = F.interpolate(segBRDFBatch, scale_factor=0.5, mode="nearest")
        notDarkEnv = (torch.mean(torch.mean(torch.mean(input_dict['envmapsBatch'], 4), 4), 1, True ) > 0.001 ).float()
        segEnvBatch = (segBRDFBatchSmall * input_dict['envmapsIndBatch'].expand_as(segBRDFBatchSmall) ).unsqueeze(-1).unsqueeze(-1)
        # print(segEnvBatch.shape, notDarkEnv.shape, segBRDFBatchSmall.shape, input_dict['envmapsIndBatch'].shape, input_dict['envmapsIndBatch'].expand_as(segBRDFBatchSmall).shape)
        # torch.Size([4, 1, 120, 160, 1, 1]) torch.Size([4, 1, 120, 160]) torch.Size([4, 1, 120, 160]) torch.Size([4, 1, 1, 1]) torch.Size([4, 1, 120, 160])

        # print(segEnvBatch.shape, notDarkEnv.shape) # torch.Size([4, 1, 120, 160, 1, 1]) torch.Size([4, 1, 120, 160])
        segEnvBatch = segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)
        
        return_dict = {}

        # Compute the recontructed error
        if self.cfg.MODEL_LIGHT.use_GT_light_envmap:
            envmapsPredImage = input_dict['envmapsBatch'].detach().clone()
        else:
            envmapsPredImage, axisPred, lambPred, weightPred = self.non_learnable_layers['output2env'].output2env(axisPred_ori, lambPred_ori, weightPred_ori, if_postprocessing=not self.cfg.MODEL_LIGHT.use_GT_light_sg)

        # print(axisPred_ori.shape, lambPred_ori.shape, weightPred_ori.shape, envmapsPredImage.shape, '=====')

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
                envmapsPredScaledImage = torch.exp(envmapsPredScaledImage_offset_log_) - self.cfg.MODEL_LIGHT.offset
            else:
                envmapsPredScaledImage = models_brdf.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(input_dict['envmapsBatch'] ),
                    input_dict['envmapsBatch'] * segEnvBatch.expand_as(input_dict['envmapsBatch']), envmapsPredImage, 
                    if_clamp_coeff=self.cfg.MODEL_LIGHT.if_clamp_coeff)
                print('-envmapsPredScaledImage-', torch.max(envmapsPredScaledImage), torch.min(envmapsPredScaledImage), torch.mean(envmapsPredScaledImage), torch.median(envmapsPredScaledImage))
                # envmapsPredScaledImage_offset_log_ = torch.log(torch.clamp(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset, min=self.cfg.MODEL_LIGHT.offset))
                envmapsPredScaledImage_offset_log_ = torch.log(envmapsPredScaledImage + self.cfg.MODEL_LIGHT.offset)

        if self.opt.is_master and if_print:
            ic(torch.max(input_dict['envmapsBatch']), torch.min(input_dict['envmapsBatch']),torch.median(input_dict['envmapsBatch']))
            ic(torch.max(envmapsPredImage), torch.min(envmapsPredImage),torch.median(envmapsPredImage))
            ic(torch.max(envmapsPredScaledImage), torch.min(envmapsPredScaledImage),torch.median(envmapsPredScaledImage))
            ic(torch.max(envmapsPredScaledImage_offset_log_), torch.min(envmapsPredScaledImage_offset_log_),torch.median(envmapsPredScaledImage_offset_log_))

        return_dict.update({'envmapsPredImage': envmapsPredImage, 'envmapsPredScaledImage': envmapsPredScaledImage, 'envmapsPredScaledImage_offset_log_': envmapsPredScaledImage_offset_log_, \
            'segEnvBatch': segEnvBatch, \
            'imBatchSmall': imBatchSmall, 'segBRDFBatchSmall': segBRDFBatchSmall, 'pixelNum_recon': pixelNum_recon}) 

        # Compute the rendered error
        pixelNum_render = max( (torch.sum(segBRDFBatchSmall ).cpu().data).item(), 1e-5 )
        
        # if not self.cfg.MODEL_LIGHT.use_GT_brdf:
        #     normal_input, rough_input = return_dict_brdf['normalInput'], return_dict_brdf['roughInput']
        # else:
        #     normal_input, rough_input = return_dict_brdf['normalInput'], return_dict_brdf['roughInput']
        normal_input, rough_input = normalInput, roughInput
        if self.cfg.DATA.if_pad_to_32x:
            normal_input = normal_input[:, :, :im_h, :im_w]
            rough_input = rough_input[:, :, :im_h, :im_w]
            albedoInput = albedoInput[:, :, :im_h, :im_w]

        if self.cfg.MODEL_LIGHT.use_GT_light_envmap:
            envmapsImage_input = input_dict['envmapsBatch']
        else:
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
        # renderedImPred = torch.clamp(renderedImPred_hdr, 0, 1)
        renderedImPred = renderedImPred_hdr

        if self.opt.is_master and if_print:
            print('--renderedImPred', torch.max(renderedImPred), torch.min(renderedImPred), torch.median(renderedImPred), renderedImPred.shape)
            # UNet: tensor(2.8596, device='cuda:0') tensor(0., device='cuda:0') tensor(0.3152, device='cuda:0') torch.Size([2, 3, 120, 160])
            # DPT: tensor(1.0586, device='cuda:0') tensor(0.0389, device='cuda:0') tensor(0.3474, device='cuda:0') torch.Size([2, 3, 120, 160])

        renderedImPred_sdr = torch.clamp(renderedImPred_hdr ** (1.0/2.2), 0, 1)

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
        albedoInput = return_dict_brdf['albedoPred'].detach().clone()
        depthInput = return_dict_brdf['depthPred'].detach().clone()
        if self.cfg.MODEL_BRDF.depth_activation == 'tanh':
            depthInput = 0.5 * (depthInput + 1) # [-1, 1] -> [0, 1]
        normalInput = return_dict_brdf['normalPred'].detach().clone()
        roughInput = return_dict_brdf['roughPred'].detach().clone()

        segBRDFBatch = input_dict['segBRDFBatch']
        pad_mask = input_dict['pad_mask'].float()

        if self.cfg.MODEL_LIGHT.DPT_baseline.enable:
            axisPred_ori, lambPred_ori, weightPred_ori = self.forward_LIGHT_Net_DPT_baseline(input_dict, imBatch, albedoInput, depthInput, normalInput, roughInput)
        else:
            axisPred_ori, lambPred_ori, weightPred_ori = self.forward_LIGHT_Net(input_dict, imBatch, albedoInput, depthInput, normalInput, roughInput)

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

        envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred_ori, lambPred_ori, weightPred_ori, if_postprocessing=not self.cfg.MODEL_LIGHT.use_GT_light_sg)

        normal_input, rough_input = normalInput, roughInput
        if self.cfg.DATA.if_pad_to_32x:
            normal_input = normal_input[:, :, :im_h, :im_w]
            rough_input = rough_input[:, :, :im_h, :im_w]
            albedoInput = albedoInput[:, :, :im_h, :im_w]

        envmapsImage_input = envmapsPredImage

        # print(im_h, im_w, normal_input.shape, envmapsImage_input.shape, albedoInput.shape, rough_input.shape, )
        diffusePred, specularPred = renderLayer.forwardEnv(normalPred=normal_input.detach(), envmap=envmapsImage_input, diffusePred=albedoInput.detach(), roughPred=rough_input.detach())

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

        cDiff, cSpec = (torch.sum(diffusePredScaled) / torch.sum(diffusePred)).data.item(), ((torch.sum(specularPredScaled) ) / (torch.sum(specularPred) ) ).data.item()
        if cSpec == 0:
            cAlbedo = 1/ axisPred_ori.max().data.item()
            cLight = cDiff / cAlbedo
        else:
            cLight = cSpec
            cAlbedo = cDiff / cLight
            cAlbedo = np.clip(cAlbedo, 1e-3, 1 / axisPred_ori.max().data.item() )
            cLight = cDiff / cAlbedo
        envmapsPredImage = envmapsPredImage * cLight
        ic(cLight, envmapsPredImage.shape, envmapsPredImage.dtype)
        ic(torch.max(envmapsPredImage), torch.min(envmapsPredImage), torch.median(envmapsPredImage), torch.mean(envmapsPredImage))

        return_dict.update({'imBatchSmall': imBatchSmall, 'envmapsPredImage': envmapsPredImage, 'renderedImPred': renderedImPred, 'renderedImPred_sdr': renderedImPred_sdr}) 
        return_dict.update({'LightNet_preds': {'axisPred': axisPred_ori, 'lambPred': lambPred_ori, 'weightPred': weightPred_ori}})

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

    def load_pretrained_MODEL_BRDF(self, if_load_encoder=True, if_load_decoder=True, if_load_Bs=False):
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
            if 'al' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['albedoDecoder']
            if 'no' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['normalDecoder']
            if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['roughDecoder']
            if 'de' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['depthDecoder']
        if if_load_Bs:
            if 'al' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['albedoBs']
            if 'no' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['normalBs']
            if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['roughBs']
            if 'de' in self.opt.cfg.MODEL_BRDF.enable_list:
                module_names += ['depthBs']
            # module_names += ['albedoBs']
            # assert self.cfg.MODEL_BRDF.if_bilateral_albedo_only

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
    
    def load_pretrained_MODEL_LIGHT(self):
        pretrained_path_root = Path(self.opt.cfg.PATH.models_ckpt_path)
        loaded_strings = []
        for saved_name in ['lightEncoder', 'axisDecoder', 'lambDecoder', 'weightDecoder', ]:
            # pickle_path = '{0}/{1}{2}_{3}.pth'.format(pretrained_path, saved_name, cascadeLevel, epochIdFineTune) 
            pickle_path = str(pretrained_path_root / self.opt.cfg.MODEL_LIGHT.pretrained_pth_name_cascade0) % saved_name
            print('Loading ' + pickle_path)
            self.LIGHT_Net[saved_name].load_state_dict(
                torch.load(pickle_path).state_dict())
            loaded_strings.append(saved_name)

            self.logger.info(magenta('Loaded pretrained LightNet-%s from %s'%(saved_name, pickle_path)))

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

    def freeze_BRDF_except_albedo(self, if_print=True):
        if 'no' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.normalDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.normalDecoder, if_print=if_print)
        if 'de' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.depthDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.depthDecoder, if_print=if_print)
        if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.roughDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.roughDecoder, if_print=if_print)

    def unfreeze_BRDF_except_albedo(self, if_print=True):
        if 'no' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.normalDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.normalDecoder, if_print=if_print)
        if 'de' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.depthDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.depthDecoder, if_print=if_print)
        if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.roughDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.roughDecoder, if_print=if_print)

    def freeze_BRDF_except_depth_normal(self, if_print=True):
        if 'al' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.albedoDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.albedoDecoder, if_print=if_print)
        if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_off_names(['BRDF_Net.roughDecoder'], if_print=if_print)
            freeze_bn_in_module(self.BRDF_Net.roughDecoder, if_print=if_print)

    def unfreeze_BRDF_except_depth_normal(self, if_print=True):
        if 'al' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.albedoDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.albedoDecoder, if_print=if_print)
        if 'ro' in self.opt.cfg.MODEL_BRDF.enable_list:
            self.turn_on_names(['BRDF_Net.roughDecoder'], if_print=if_print)
            unfreeze_bn_in_module(self.BRDF_Net.roughDecoder, if_print=if_print)


    def freeze_bn_semantics(self):
        freeze_bn_in_module(self.SEMSEG_Net)

    def freeze_bn_matseg(self):
        freeze_bn_in_module(self.MATSEG_Net)