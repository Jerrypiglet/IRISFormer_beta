import torch
import torch.nn as nn

from models_def.model_matseg import Baseline
import models_def.models_brdf as models_brdf
from utils.utils_misc import *
import pac
from utils.utils_training import freeze_bn_in_module
import torch.nn.functional as F

class BRDFplusSemSeg(nn.Module):
    def __init__(self, opt, logger):
        super(BRDFplusSemSeg, self).__init__()
        self.opt = opt
        self.cfg = opt.cfg
        self.logger = logger

        # if self.cfg.MODEL_SEMSEG.enable:
        #     value_scale = 255
        #     mean = [0.485, 0.456, 0.406]
        #     mean = [item * value_scale for item in mean]
        #     self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(opt.device)
        #     std = [0.229, 0.224, 0.225]
        #     std = [item * value_scale for item in std]
        #     self.std = torch.tensor(std).view(1, 3, 1, 1).to(opt.device)
        #     self.semseg_configs = self.opt.semseg_configs
        #     self.semseg_path = self.semseg_configs.semseg_path_cluster if opt.if_cluster else self.semseg_configs.semseg_path_local
        #     # self.UNet = Baseline(self.cfg.MODEL_SEMSEG)
        #     assert self.semseg_configs.arch == 'psp'

        #     from models_def.models_semseg.pspnet import PSPNet
        #     self.SEMSEG_Net = PSPNet(layers=self.semseg_configs.layers, classes=self.semseg_configs.classes, zoom_factor=self.semseg_configs.zoom_factor, pretrained=False)
        #     self.SEMSEG_Net.eval()

        #     if self.opt.cfg.MODEL_SEMSEG.load_pretrained_pth:
        #         self.load_pretrained_semseg()
        
        if self.cfg.MODEL_BRDF.enable:
            self.BRDF_Net = nn.ModuleDict(
                {
                    'encoder': models_brdf.encoder0(opt, cascadeLevel = self.opt.cascadeLevel, in_channels = 3 if not self.opt.ifMatMapInput else 4), 
                    'albedoDecoder': models_brdf.decoder0(opt, mode=0), 
                    'normalDecoder': models_brdf.decoder0(opt, mode=1), 
                    'roughDecoder': models_brdf.decoder0(opt, mode=2), 
                    'depthDecoder': models_brdf.decoder0(opt, mode=4)
                }
            )
        if self.cfg.MODEL_BRDF.enable_semseg_decoder:
            self.BRDF_Net.update({'semsegDecoder': models_brdf.decoder0(opt, mode=-1, out_channel=self.cfg.DATA.semseg_classes, if_PPM=self.cfg.MODEL_BRDF.semseg_PPM)})
            # self.BRDF_Net.update({'semsegDecoder': models_brdf.decoder0(opt, mode=-1, out_channel=self.cfg.MODEL_BRDF.semseg_classes, \
                # in_C=[1024, 1024, 512, 512, 256, 192], out_C=[512, 256, 256, 128, 128, 128], group_C=[32, 16, 16, 8, 8, 8], if_PPM=self.cfg.MODEL_BRDF.semseg_PPM)})
        # self.guide_net = guideNet(opt)

    # def forward_mat_seg(self, input_dict):
    #     return self.UNet(input_dict['im_batch'])

    # def forward_semseg(self, input_dict):
    #     im_batch_255 = input_dict['im_uint8']
    #     im_batch_255_float = input_dict['im_uint8'].float().permute(0, 3, 1, 2)
    #     im_batch_255_float = F.pad(im_batch_255_float, (0, 1, 0, 1))
    #     # print(im_batch_255_float.shape, im_batch_255_float_padded.shape)
    #     # print(torch.max(im_batch_255_float), torch.min(im_batch_255_float), torch.median(im_batch_255_float), im_batch_255_float.shape, self.mean.shape)
    #     im_batch_255_float.sub_(self.mean).div_(self.std)
    #     # print(torch.max(im_batch_255_float), torch.min(im_batch_255_float), torch.median(im_batch_255_float))
    
    #     self.SEMSEG_Net.eval()
    #     with torch.no_grad():
    #         output_dict_PSPNet = self.SEMSEG_Net(im_batch_255_float)
    #     output_PSPNet = output_dict_PSPNet['x']
    #     feat_dict_PSPNet = output_dict_PSPNet['feat_dict']
    
    #     # print(output_PSPNet.shape)

    #     _, _, h_i, w_i = im_batch_255_float.shape
    #     _, _, h_o, w_o = output_PSPNet.shape
    #     # print(h_o, h_i, w_o, w_i)
    #     assert (h_o == h_i) and (w_o == w_i)
    #     # if (h_o != h_i) or (w_o != w_i):
    #         # output_PSPNet = F.interpolate(output_PSPNet, (h_i, w_i), mode='bilinear', align_corners=True)
    #     output_PSPNet = output_PSPNet[:, :, :-1, :-1]
    #     output_PSPNet_softmax = F.softmax(output_PSPNet, dim=1)

    #     return_dict = {'output_PSPNet': output_PSPNet, 'output_PSPNet_softmax': output_PSPNet_softmax}
    #     return_dict.update({'feats_semseg_dict': feat_dict_PSPNet})

    #     return return_dict


    def forward_brdf(self, input_dict, input_dict_guide=None):
        # x1, x2, x3, x4, x5, x6 = self.BRDF_Net['encoder'](input_dict['input_batch_brdf']) # [16, 64, 96, 128], [16, 128, 48, 64], [16, 256, 24, 32], [16, 256, 12, 16], [16, 512, 6, 8]
        # albedoPred = 0.5 * (self.BRDF_Net['albedoDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_guide=input_dict_guide) + 1)

            # Initial Prediction
        x1, x2, x3, x4, x5, x6 = self.BRDF_Net['encoder'](input_dict['input_batch_brdf'])
        albedoPred = 0.5 * (self.BRDF_Net['albedoDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_guide=input_dict_guide) + 1)
        normalPred = self.BRDF_Net['normalDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_guide=input_dict_guide)
        roughPred = self.BRDF_Net['roughDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_guide=input_dict_guide)
        depthPred = 0.5 * (self.BRDF_Net['depthDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6, input_dict_guide=input_dict_guide) + 1)
        # print(torch.min(depthPred), torch.max(depthPred), torch.mean(depthPred), torch.median(depthPred))

        # if not self.opt.test_real:
        input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
        albedoPred = models_brdf.LSregress(albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred),
                input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoPred)
        albedoPred = torch.clamp(albedoPred, 0, 1)

        depthPred = models_brdf.LSregress(depthPred *  input_dict['segAllBatch'].expand_as(depthPred),
                input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthPred)

        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape)
        return_dict = {'albedoPred': albedoPred, 'normalPred': normalPred, 'roughPred': roughPred, 'depthPred': depthPred}

        if self.cfg.MODEL_BRDF.enable_semseg_decoder:
            semsegPred = self.BRDF_Net['semsegDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6)
            return_dict.update({'semseg_pred': semsegPred})

        return return_dict
    
    def forward(self, input_dict):
        return_dict = {}
        # if self.cfg.MODEL_SEG.enable:
        #     return_dict_mat_seg = self.forward_mat_seg(input_dict) # {'prob': prob, 'embedding': embedding, 'feats_mat_seg_dict': feats_mat_seg_dict}
        #     input_dict_guide = return_dict_mat_seg['feats_mat_seg_dict']
        # else:
        #     return_dict_mat_seg = {}
        #     input_dict_guide = None

        if self.cfg.MODEL_SEMSEG.enable:
            return_dict_semseg = self.forward_semseg(input_dict) # {'prob': prob, 'embedding': embedding, 'feats_mat_seg_dict': feats_mat_seg_dict}
            input_dict_guide = return_dict_semseg['feats_semseg_dict']
        else:
            return_dict_semseg = {}
            input_dict_guide = None
        return_dict.update(return_dict_semseg)

        # return_dict_guide = self.guide_net(return_dict_mat_seg['feats_mat_seg_dict'])

        if self.cfg.MODEL_BRDF.enable:
            return_dict_brdf = self.forward_brdf(input_dict, input_dict_guide=input_dict_guide)
        else:
            return_dict_brdf = {}
        return_dict.update(return_dict_brdf)
        
        return return_dict

    def print_net(self):
        count_grads = 0
        for name, param in self.named_parameters():
            if_trainable_str = white_blue('True') if param.requires_grad else green('False')
            self.logger.info(name + str(param.shape) + if_trainable_str)
            if param.requires_grad:
                count_grads += 1
        self.logger.info(magenta('---> ALL %d params; %d trainable'%(len(list(self.named_parameters())), count_grads)))
        return count_grads

    def load_pretrained_brdf(self, pretrained_brdf_name='check_cascade0_w320_h240'):
        if self.opt.if_cluster:
            pretrained_path = '/viscompfs/users/ruizhu/models_ckpt/' + pretrained_brdf_name
        else:
            pretrained_path = '/home/ruizhu/Documents/Projects/semanticInverse/models_ckpt/' + pretrained_brdf_name
        self.logger.info(magenta('Loading pretrained BRDF model from %s'%pretrained_path))
        cascadeLevel = 0
        epochIdFineTune = 13
        print('{0}/encoder{1}_{2}.pth'.format(pretrained_path, cascadeLevel, epochIdFineTune))
        print(torch.load('{0}/encoder{1}_{2}.pth'.format(pretrained_path, cascadeLevel, epochIdFineTune) ))
        self.BRDF_Net['encoder'].load_state_dict(
            torch.load('{0}/encoder{1}_{2}.pth'.format(pretrained_path, cascadeLevel, epochIdFineTune) ).state_dict())
        self.BRDF_Net['albedoDecoder'].load_state_dict(
                torch.load('{0}/albedo{1}_{2}.pth'.format(pretrained_path, cascadeLevel, epochIdFineTune) ).state_dict() )
        self.BRDF_Net['normalDecoder'].load_state_dict(
                torch.load('{0}/normal{1}_{2}.pth'.format(pretrained_path, cascadeLevel, epochIdFineTune) ).state_dict() )
        self.BRDF_Net['roughDecoder'].load_state_dict(
                torch.load('{0}/rough{1}_{2}.pth'.format(pretrained_path, cascadeLevel, epochIdFineTune) ).state_dict() )
        self.BRDF_Net['depthDecoder'].load_state_dict(
                torch.load('{0}/depth{1}_{2}.pth'.format(pretrained_path, cascadeLevel, epochIdFineTune) ).state_dict() )

    def load_pretrained_semseg(self):
        self.print_net()
        model_path = self.semseg_path + self.semseg_configs.model_path
        if os.path.isfile(model_path):
            self.logger.info(red("=> loading checkpoint '{}'".format(model_path)))
            state_dict = torch.load(model_path)['state_dict']
            # print(state_dict.keys())
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            replace_dict = {'layer0.0': 'layer0_1.0', 'layer0.1': 'layer0_1.1', 'layer0.3': 'layer0_2.0', 'layer0.4': 'layer0_2.1', 'layer0.6': 'layer0_3.0', 'layer0.7': 'layer0_3.1'}
            state_dict = {k.replace(key, replace_dict[key]): v for k, v in state_dict.items() for key in replace_dict}
            
            self.SEMSEG_Net.load_state_dict(state_dict)
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
                    self.logger.info(colored('turn_ON_in_names: ' + in_name, 'white', 'on_red'))


    def turn_off_names(self, in_names):
        for name, param in self.named_parameters():
            for in_name in in_names:
            # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if in_name in name:
                    param.requires_grad = False
                    self.logger.info(colored('turn_False_in_names: ' + in_name, 'white', 'on_red'))

    def freeze_bn_semantics(self):
        freeze_bn_in_module(self.SEMSEG_Net)

# class guideNet(nn.Module):
#     def __init__(self, opt):
#         super(guideNet, self).__init__()
#         self.opt = opt
#         self.guide_C = self.opt.cfg.MODEL_SEG.guide_channels
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


