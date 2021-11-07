# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DTYPE = "float32"

_C.MM1_DEBUG = False

_C.PATH = CN()
_C.PATH.cluster_names = ['kubectl', 'nvidia', 'ngc']
_C.PATH.root = ''
_C.PATH.root_local = '/home/ruizhu/Documents/Projects/semanticInverse/train'
_C.PATH.root_cluster = ['.', '.', '.']
_C.PATH.semseg_colors_path = 'data/openrooms/openrooms_colors.txt'
_C.PATH.semseg_names_path = 'data/openrooms/openrooms_names.txt'
_C.PATH.total3D_colors_path = 'data/openrooms/total3D_colors'

_C.PATH.total3D_lists_path = 'data/openrooms/list_OR_V4full'
_C.PATH.total3D_lists_path_if_zhengqinCVPR = False
_C.PATH.total3D_lists_path_zhengqinCVPR = 'data/openrooms/list_OR_V4full_zhengqinCVPR'

_C.PATH.matcls_matIdG1_path = 'data/openrooms/matIdGlobal1.txt'
_C.PATH.matcls_matIdG2_path = 'data/openrooms/matIdGlobal2.txt'
_C.PATH.dcn_path = ''
_C.PATH.dcn_cluster = ['/viscompfs/users/ruizhu/lib/Deformable-Convolution-V2-PyTorch', '', '']
_C.PATH.dcn_local = '/home/ruizhu/Documents/Projects/semanticInverse/third-partieresults_outside/Deformable-Convolution-V2-PyTorch'
_C.PATH.torch_home_path = ''
_C.PATH.torch_home_local = '/home/ruizhu/Documents/Projects/semanticInverse/'
_C.PATH.torch_home_cluster = ['/ruidata/semanticInverse/', '/home/ruzhu/Documents/torch', '/newfoundland/torch']
# _C.DATA.semseg_colors_path = 'data/openrooms/openrooms_colors.txt'
# _C.DATA.semseg_names_path = 'data/openrooms/openrooms_names.txt'
_C.PATH.OR4X_mapping_catInt_to_RGB = ['data/openrooms/total3D_colors/OR4X_mapping_catInt_to_RGB_light.pkl', 'data/openrooms/total3D_colors/OR4X_mapping_catInt_to_RGB_dark.pkl']
_C.PATH.OR4X_mapping_catStr_to_RGB = ['data/openrooms/total3D_colors/OR4X_mapping_catStr_to_RGB_light.pkl', 'data/openrooms/total3D_colors/OR4X_mapping_catStr_to_RGB_dark.pkl']
_C.PATH.pretrained_path = ''
_C.PATH.pretrained_local = '/home/ruizhu/Documents/Projects/semanticInverse/pretrained'
_C.PATH.pretrained_cluster = ['/ruidata/semanticInverse/pretrained', '/home/ruzhu/Documents/Projects/semanticInverse/pretrained', '/newfoundland/semanticInverse/pretrained/']

# ===== debug

_C.DEBUG = CN()
_C.DEBUG.if_fast_BRDF_labels = True
_C.DEBUG.if_fast_light_labels = True
_C.DEBUG.if_dump_anything = False
_C.DEBUG.if_test_real = False
_C.DEBUG.if_iiw = False
_C.DEBUG.if_dump_shadow_renderer = False
_C.DEBUG.if_dump_perframe_BRDF = False

_C.DEBUG.dump_BRDF_offline = CN()
_C.DEBUG.dump_BRDF_offline.enable = False
_C.DEBUG.dump_BRDF_offline.path_root = ''
_C.DEBUG.dump_BRDF_offline.path_root_local = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_dump'
_C.DEBUG.dump_BRDF_offline.path_root_cluster = ['/ruidata/openrooms_dump', '', '']
_C.DEBUG.dump_BRDF_offline.path_task = ''
_C.DEBUG.dump_BRDF_offline.task_name = 'BRDFmodel'
_C.DEBUG.if_load_dump_BRDF_offline = False

# ===== dataset

_C.DATASET = CN()
_C.DATASET.mini = False # load mini OR from SSD to enable faster dataloading for debugging purposes etc.
_C.DATASET.tmp = False # load tmp OR list from DATASET.dataset_list_tmp
_C.DATASET.first_scenes = -1 # laod first # of the entire dataset: train/val
_C.DATASET.dataset_name = 'openrooms'
_C.DATASET.dataset_path = ''
_C.DATASET.dataset_path_local = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms'
_C.DATASET.dataset_path_local_quarter = '/ruidata/openrooms_raw_quarter'
_C.DATASET.dataset_path_cluster = ['/siggraphasia20dataset/code/Routine/DatasetCreation/', '', '/datasets_mount/']
_C.DATASET.dataset_path_binary = ''
_C.DATASET.dataset_path_local_fast_BRDF = '/ruidata/openrooms_raw_BRDF'

_C.DATASET.real_images_root_path = '/home/ruizhu/Documents/Projects/semanticInverse'
_C.DATASET.real_images_list_path = 'data/list_real_20.txt'

_C.DATASET.iiw_path = ''
_C.DATASET.iiw_path_local = '/ruidata/iiw-dataset/data'
_C.DATASET.iiw_path_cluster = ['', '', '']
_C.DATASET.iiw_list_path = 'data/iiw/list'

_C.DATASET.dataset_path_binary_local = '/newfoundland2/ruizhu/ORfull-seq-240x320'
# _C.DATASET.dataset_path_binary_local = '/newfoundland2/ruizhu/ORfull-seq-240x320-albedoInOneFile'
_C.DATASET.dataset_path_binary_cluster = ['/ruidata/ORfull-seq-240x320-smaller-RE', '', '/datasets_mount/ORfull-seq-240x320-smaller-RE']
_C.DATASET.dataset_path_binary_root = '/datasets_mount'

_C.DATASET.dataset_path_pickle = ''
_C.DATASET.dataset_path_pickle_local = '/newfoundland2/ruizhu/ORfull-perFramePickles-240x320'
_C.DATASET.dataset_path_pickle_cluster = ['/ruidata/ORfull-perFramePickles-240x320', '', '/datasets_mount/ORfull-perFramePickles-240x320']

# _C.DATASET.dataset_path_binary_cluster = ['/ruidata/ORfull-seq-240x320-albedoInOneFile', '/local/ruzhu/data/ORfull-seq-240x320', '/datasets_mount/ORfull-seq-240x320-albedoInOneFile']
_C.DATASET.dataset_path_test = ''
_C.DATASET.dataset_path_test_local = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_test'
_C.DATASET.dataset_path_test_cluster = ['/eccv20dataset/DatasetNew_test', '', '']
_C.DATASET.png_path = ''
_C.DATASET.png_path_local = '/data/ruizhu/OR-pngs'
# _C.DATASET.png_path_local = ''
_C.DATASET.png_path_cluster = ['/siggraphasia20dataset/pngs', '/local/ruzhu/data/OR-pngs', '/datasets_mount/OR-pngs']

_C.DATASET.swin_path = ''
_C.DATASET.swin_path_local = '/home/ruizhu/Documents/Projects/semanticInverse/third_parties_outside/Swin-Transformer-Semantic-Segmentation'
_C.DATASET.swin_path_cluster = ['/ruidata/semanticInverse/third_parties_outside/Swin-Transformer-Semantic-Segmentation', '', '']

_C.DATASET.if_to_memory = False
_C.DATASET.memory_path = '/dev/shm'

_C.DATASET.if_binary = False # load binary version of dataset instead of from per-sample files
_C.DATASET.binary = CN()
_C.DATASET.binary.if_in_one_file = False
_C.DATASET.binary.if_shuffle = False

_C.DATASET.if_pickle = False
_C.DATASET.pickle = CN()

_C.DATASET.layout_emitter_path = ''
# _C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-detachEmitter-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-detachEmitterRERE-OR45_total3D_train_test_data'

# _C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-withMesh20210507-val500-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-withMesh20210507-val-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-withMesh20210507-OR45_total3D_train_test_data'
_C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-withMesh20210510-assign2NotReindex-OR45_total3D_train_test_data'

# _C.DATASET.layout_emitter_path_cluster = '/ruidata/OR-V4full-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_cluster = '/ruidata/OR-V4full-detachEmitter-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_cluster = '/ruidata/OR-V4full-detachEmitterRERE-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_cluster = '/ruidata/OR-V4full-detachEmitterRERERE20210502-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_cluster = '/ruidata/OR-V4full-withMesh20210507-OR45_total3D_train_test_data'
_C.DATASET.layout_emitter_path_cluster = ['/ruidata/OR-V4full-withMesh20210510-assign2NotReindex-OR45_total3D_train_test_data_3', '', '']

_C.DATASET.envmap_path = ''
_C.DATASET.envmap_path_local = '/home/ruizhu/Documents/data/EnvDataset/'
_C.DATASET.envmap_path_cluster = ['/siggraphasia20dataset/EnvDataset/', '', '']

_C.DATASET.matpart_path = ''
_C.DATASET.matpart_path_local = '/data/ruizhu/OR-matpart'
_C.DATASET.matpart_path_cluster = ['/siggraphasia20dataset/code/Routine/DatasetCreation/', '', '']
_C.DATASET.matori_path = ''
_C.DATASET.matori_path_local = '/newfoundland2/ruizhu/siggraphasia20dataset/BRDFOriginDataset/'
_C.DATASET.matori_path_cluster = ['/siggraphasia20dataset/BRDFOriginDataset/', '', '']

# _C.DATASET.dataset_list = 'data/openrooms/list_OR_V4full/list'
_C.DATASET.dataset_list = ''
_C.DATASET.dataset_path_mini = ''
_C.DATASET.dataset_path_mini_local = '/data/ruizhu/openrooms_mini'
_C.DATASET.dataset_path_mini_cluster = ['/ruidata/openrooms_mini', '/local/ruzhu/data/openrooms_mini', '/datasets_mount/openrooms_mini']
_C.DATASET.dataset_path_mini_binary = ''
_C.DATASET.dataset_path_mini_binary_local = '/home/ruizhu/Documents/data/OR-seq-mini-240x320'
_C.DATASET.dataset_path_mini_binary_cluster = ['', '', '']
_C.DATASET.dataset_path_mini_pickle = ''
_C.DATASET.dataset_path_mini_pickle_local = '/home/ruizhu/Documents/data/OR-perFramePickles-mini-240x320'
_C.DATASET.dataset_path_mini_pickle_cluster = ['/ruidata/ORfull-perFramePickles-240x320', '', '/datasets_mount/ORmini-perFramePickles-240x320']
_C.DATASET.dataset_list_mini = 'data/openrooms/list_ORmini/list'
# _C.DATASET.dataset_path_mini = '/data/ruizhu/openrooms_mini-val'
# _C.DATASET.dataset_list_mini = 'data/openrooms/list_ORmini-val/list'
_C.DATASET.dataset_path_tmp = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_sequence_val_skip20Frames_withDepth'
_C.DATASET.dataset_list_tmp = 'data/openrooms/list_OR_scanNetPose_hasNext/list'
# _C.DATASET.dataset_path_tmp = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_sequence_val_notSkipFrames_withDepth_tmp'
# _C.DATASET.dataset_list_tmp = 'data/openrooms/list_OR_tmp_/list'
_C.DATASET.dataset_if_save_space = True # e.g. only same one depth for main_xml, diffMat, diffLight

_C.DATASET.dataset_list_sequence = False # convert #idx of the val list into sequential inputs
_C.DATASET.dataset_list_sequence_idx = -1

_C.DATASET.num_workers = 8
_C.DATASET.if_val_dist = True
_C.DATASET.if_no_gt_semantics = False
_C.DATASET.if_quarter = False

_C.DATASET.if_no_gt_BRDF = False
_C.DATASET.if_no_gt_light = False


# ===== data loading configs

_C.DATA = CN()
_C.DATA.if_load_png_not_hdr = False # load png as input image instead of hdr image
_C.DATA.if_also_load_next_frame = False # load next frame (only png supported) in addition to current frame

_C.DATA.if_augment_train = False
_C.DATA.im_height = 240
_C.DATA.im_width = 320
_C.DATA.im_height_padded = 256
_C.DATA.im_width_padded = 320
_C.DATA.im_height_ori = 480
_C.DATA.im_width_ori = 640
_C.DATA.if_pad_to_32x = False # if pad both height and width to multplicative of 32 (for DPT)
_C.DATA.pad_option = 'const' # [const, reflect]
_C.DATA.if_resize_to_32x = False # does not work for semantics (e.g. matseg)
_C.DATA.load_semseg_gt = False
_C.DATA.load_matseg_gt = False
_C.DATA.load_brdf_gt = True
_C.DATA.load_masks = False
_C.DATA.load_light_gt = False
_C.DATA.load_layout_emitter_gt = False
_C.DATA.data_read_list = ''
_C.DATA.data_read_list_allowed = ['al', 'no', 'de', 'ro', 'li', \
    'lo', 'em', 'ob', 'mesh']
_C.DATA.load_matcls_gt = False
_C.DATA.load_detectron_gt = False
_C.DATA.load_cam_pose = False

_C.DATA.iiw = CN()
_C.DATA.iiw.im_height = 341
_C.DATA.iiw.im_width = 512
# _C.DATA.iiw.im_height_padded = 352
# _C.DATA.iiw.im_width_padded = 512
_C.DATA.iiw.im_height_padded = 256
_C.DATA.iiw.im_width_padded = 320


# ===== GMM
_C.MODEL_GMM = CN()
_C.MODEL_GMM.enable = False
_C.MODEL_GMM.appearance_recon = CN() # single frame only
_C.MODEL_GMM.appearance_recon.enable = False
_C.MODEL_GMM.appearance_recon.sanity_check = False
_C.MODEL_GMM.appearance_recon.modalities = 'al'
_C.MODEL_GMM.feat_recon = CN() # single frame only
_C.MODEL_GMM.feat_recon.enable = False
_C.MODEL_GMM.feat_recon.n_iter = 10
_C.MODEL_GMM.feat_recon.layers_list = 'dx3_dx4_dx5' # 'dx3_dx4_dx5_x1_x2_x3'
_C.MODEL_GMM.feat_recon.use_matseg = False
_C.MODEL_GMM.feat_recon.matseg_H = 6
_C.MODEL_GMM.feat_recon.matseg_W = 8

_C.MODEL_GMM.learning_rate = 1e-5
_C.MODEL_GMM.ssn_grid_spixel = False
_C.MODEL_GMM.src_idx = 1
_C.MODEL_GMM.grad_clip = 0.
_C.MODEL_GMM.RAFT = CN()
_C.MODEL_GMM.RAFT.small = False
_C.MODEL_GMM.RAFT.OF_model_path = '/home/ruizhu/Documents/Projects/nvidia/vidapp/third-party/raft/models/raft-things.pth'

# ===== BRDF
_C.MODEL_BRDF = CN()
_C.MODEL_BRDF.enable = False
_C.MODEL_BRDF.if_bilateral = False
_C.MODEL_BRDF.if_bilateral_albedo_only = False
_C.MODEL_BRDF.if_freeze = False
# _C.MODEL_BRDF.enable_list = ['al', 'no', 'de', 'ro', 'li']
_C.MODEL_BRDF.enable_list = '' # `al_no_de_ro`
_C.MODEL_BRDF.enable_list_allowed = ['al', 'no', 'de', 'ro']
_C.MODEL_BRDF.load_pretrained_pth = False
_C.MODEL_BRDF.loss_list = ''
_C.MODEL_BRDF.channel_multi = 1
_C.MODEL_BRDF.albedoWeight = 1.5
_C.MODEL_BRDF.normalWeight = 1.0
_C.MODEL_BRDF.roughWeight = 0.5
_C.MODEL_BRDF.depthWeight = 0.5
_C.MODEL_BRDF.if_debug_arch = False
_C.MODEL_BRDF.enable_BRDF_decoders = False
# _C.MODEL_BRDF.is_all_light = True
_C.MODEL_BRDF.enable_semseg_decoder = False
_C.MODEL_BRDF.semseg_PPM = True

# _C.MODEL_BRDF.pretrained_pth_name = 'check_cascade0_w320_h240/%s0_13.pth' # should not use for Rui's splits; this ckpt was trained with Zhengqin's CVPR'20 splits
_C.MODEL_BRDF.pretrained_pth_name = 'checkBs_cascade0_w320_h240/%s0_14_1000.pth' # should not use for Rui's splits; this ckpt was trained with Zhengqin's CVPR'20 splits
_C.MODEL_BRDF.pretrained_if_load_encoder = True
_C.MODEL_BRDF.pretrained_if_load_decoder = True
_C.MODEL_BRDF.pretrained_if_load_Bs = True

_C.MODEL_BRDF.encoder_exclude = '' # e.g. 'x4_x5
_C.MODEL_BRDF.use_scale_aware_albedo = True # [default: False] set to False to use **scale-invariant** loss for albedo
_C.MODEL_BRDF.loss = CN()
_C.MODEL_BRDF.loss.if_use_reg_loss_depth = False
_C.MODEL_BRDF.loss.reg_loss_depth_weight = 0.5
_C.MODEL_BRDF.loss.if_use_reg_loss_albedo = False
_C.MODEL_BRDF.loss.reg_loss_albedo_weight = 0.5

_C.MODEL_BRDF.use_scale_aware_depth = True
_C.MODEL_BRDF.depth_activation = 'relu'
_C.MODEL_BRDF.loss.depth = CN() # ONLY works for MODEL_ALL (DPT) for now
_C.MODEL_BRDF.loss.depth.if_use_midas_loss = False # DPT: scale-invariant loss on inv depth; relu
_C.MODEL_BRDF.loss.depth.if_use_paper_loss = False # log(depth+0.001) instead of log(depth+1)
_C.MODEL_BRDF.loss.depth.if_use_Zhengqin_loss = True # tanh + loss on depth

_C.MODEL_BRDF.DPT_baseline = CN()
_C.MODEL_BRDF.DPT_baseline.enable = False
_C.MODEL_BRDF.DPT_baseline.if_share_patchembed = False
_C.MODEL_BRDF.DPT_baseline.if_share_pretrained = False
_C.MODEL_BRDF.DPT_baseline.if_share_decoder_over_heads = True # [always True] used in constructing MODEL_ALL (ViT); currently treating each modality (e.g. albedo, roughness) as one decoder with one head

_C.MODEL_BRDF.DPT_baseline.if_SGD = False
_C.MODEL_BRDF.DPT_baseline.if_pos_embed = False
_C.MODEL_BRDF.DPT_baseline.if_batch_norm = True # in DPT output head
_C.MODEL_BRDF.DPT_baseline.if_batch_norm_depth_override = True # in DPT output head
_C.MODEL_BRDF.DPT_baseline.if_vis_CA_proj_coef = False
_C.MODEL_BRDF.DPT_baseline.if_vis_CA_SSN_affinity = False
_C.MODEL_BRDF.DPT_baseline.if_vis_CA_SSN_gt_matseg = False
# _C.MODEL_BRDF.DPT_baseline.if_batch_norm_in_proj_extra = False
_C.MODEL_BRDF.DPT_baseline.modality = 'enabled'
_C.MODEL_BRDF.DPT_baseline.model = 'dpt_hybrid'
_C.MODEL_BRDF.DPT_baseline.readout = 'project'
_C.MODEL_BRDF.DPT_baseline.use_vit_only = False
# _C.MODEL_BRDF.DPT_baseline.dpt_hybrid_path = 'dpt_weights/dpt_hybrid-midas-501f0c75.pt'
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid_path = 'NA'
# _C.MODEL_BRDF.DPT_baseline.dpt_SSN_path = 'NA'
_C.MODEL_BRDF.DPT_baseline.dpt_base_path = 'NA'
# _C.MODEL_BRDF.DPT_baseline.dpt_large_path = 'dpt_weights/dpt_large-midas-2f21e586.pt'
_C.MODEL_BRDF.DPT_baseline.dpt_large_path = 'NA'
_C.MODEL_BRDF.DPT_baseline.if_freeze_backbone = False
_C.MODEL_BRDF.DPT_baseline.if_enable_attention_hooks = False
_C.MODEL_BRDF.DPT_baseline.if_freeze_pretrained = False
_C.MODEL_BRDF.DPT_baseline.if_imagenet_backbone = True
_C.MODEL_BRDF.DPT_baseline.if_skip_last_conv = True
_C.MODEL_BRDF.DPT_baseline.if_skip_patch_embed_proj = False
_C.MODEL_BRDF.DPT_baseline.if_only_restore_backbone = False
_C.MODEL_BRDF.DPT_baseline.if_batch_norm_in_proj_extra_in_proj_extra = False
_C.MODEL_BRDF.DPT_baseline.if_simple_proj_extra = False
_C.MODEL_BRDF.DPT_baseline.feat_proj_channels = -1

_C.MODEL_BRDF.DPT_baseline.patch_size = 16

_C.MODEL_BRDF.DPT_baseline.dpt_hybrid = CN()
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.N_layers = -1 # only support 4 outout layers to avoid drastic changes to original DPT
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.dual_lr = False # faster: 1e-4, backbone: 1e-5
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.yogo_lr = False # use yogo scheduler and optimizer
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.feat_proj_channels = 768

_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.depth = CN()
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.depth.activation = 'tanh'


_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA = CN()
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA = False # use Cross Attention instead of assembling
# _C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA_if_recompute_C = False # recompute tokens from im_feat then feed to next transformer; not applicable to CAv2
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_init_img_feat = False
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CA_if_grid_assembling = False # reverting back to grid assembling in CA
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.im_feat_init_c = None
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_res_CA_except_first = False
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_type = 'full' # [single, double, full] of resnet
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_full = CN()
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_shared_stem = False # if shared one resnet; or separate for tokens and im_feats
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_full.backbone_dims = 1856 # 64+256+512+1024
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_full.proj_extra_dims = 768 # to be consistent with hybrid
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_full.proj_extra_if_inst_norm = True
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.stem_full.proj_extra_if_simple = False

_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_CAc = False # use Cross Attention-tokens to update tokens (proj im feat to tokens)
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.CAc = CN()
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.CAc.if_use_previous_feat = False # use previous im_feat instead of current im_feat
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.CAc.if_use_init_feat = False # use previous im_feat instead of current im_feat

_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_SSN = False # use SSN to generate initial token
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN = CN()
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_freeze_matseg = True
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_hard_affinity_for_c = False
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_dense = False
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_gt_matseg = False # if use GT matseg labels as hard affinity
# _C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_inject_gt_affinity = False # if inject (multiply) gt affinity (binary) to CA
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_gt_matseg_if_duplicate_tokens = False # if duplicate tokens to avoid empty tokens instead of applying token masks to everywhere
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_gt_matseg_if_inject_token_mask = False # if inject (multiply) gt token masks (binary) to CA for reconstruction
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_gt_matseg_if_inject_proj_coef = False
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_gt_matseg_if_inject_transformer = False
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.ssn_from = 'matseg' # ['backbone', 'matseg', 'matseg-2']
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.backbone_dims = 1856 # 64+256+512+1024
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.proj_extra_dims = 768 # to be consistent with hybrid
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.proj_extra_if_inst_norm = True
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.proj_extra_if_simple = False
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_ssn_matseg_on_lower_res = False # if SSN on quarter matseg feat instead of on full res and then resize

_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_not_reduce_res = False # if not reducing resolution in vit_CAv2.py
_C.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_only_last_transformer_output_used = False # [debug] if only using the last transformer outoput in unet, instead of using 4 intermediate outputs

_C.MODEL_BRDF.DPT_baseline.dpt_large = CN()
_C.MODEL_BRDF.DPT_baseline.dpt_large.feat_proj_channels = 1024

_C.MODEL_BRDF.DPT_baseline.dpt_SSN = CN() # dpt_hybrid_SSN (V1)
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_from = 'matseg' # ['backbone', 'matseg']
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.ssn_recon_method = 'qkv' # ['qtc', 'qkv']
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.keep_N_layers = -1 # set to -1 to disable; otherwise denoting keeping the first N layers
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv = False # if recompute im_feat and codebook bwtween transformer layers  to qkv mode; similar to YOGO
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_slim = False # if recompute im_feat and codebook bwtween transformer layers  to qkv mode; similar to YOGO
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_recompute_C = False # if not recompute C by QI_new
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_not_reduce_res = False # if not reducing resolution in vit_SSN_unet_qkv.py
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_only_last_transformer_output_used = False # [debug] if only using the last transformer outoput in unet, instead of using 4 intermediate outputs
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_use_init_img_feat = False
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_use_Q_as_proj_coef = False
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_res_add_after_QI = False # if not reducing resolution in vit_SSN_unet_qkv.py
# _C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_if_vis_affinity_in_layers = False # if compute affinity between tokens and im
# _C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_transform_feat_in_qkv_reconQ_from = 'proj_coef' # ['proj_coef', 'affinity']
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_perpixel_abs_pos_embed = False
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_proj_method = 'full' # if using residual/concat/none in projector in CrossAttention module
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.feat_fusion_method = 'sum' # fusion method in class FeatureFusionBlock_custom
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.ca_norm_layer = 'instanceNorm' # [instanceNorm, identity, layerNorm]
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.backbone_dims = 1344 # resnet: stem + stage 0,1,2
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_freeze_matseg = True
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_dense = True
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_freeze_unet = False
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_debug_unet = False
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone = False # use unet instead of resnet as backbone, for DPT-ssn models
_C.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_feat_in_transformer = False # use unet instead of resnet as backbone, for DPT-ssn models
# _C.MODEL_BRDF.DPT_baseline.dpt_SSN.spixel_h = 12
# _C.MODEL_BRDF.DPT_baseline.dpt_SSN.spixel_w = 16

# ===== per-pixel lighting
_C.MODEL_LIGHT = CN()
_C.MODEL_LIGHT.enable = False
_C.MODEL_LIGHT.if_freeze = False
_C.MODEL_LIGHT.envRow = 120
_C.MODEL_LIGHT.envCol = 160
_C.MODEL_LIGHT.envHeight = 8
_C.MODEL_LIGHT.envWidth = 16
_C.MODEL_LIGHT.SGNum = 12
_C.MODEL_LIGHT.envmapWidth = 1024
_C.MODEL_LIGHT.envmapHeight = 512
_C.MODEL_LIGHT.offset = 1. # 'the offset for log error'
_C.MODEL_LIGHT.use_GT_brdf = False
_C.MODEL_LIGHT.use_offline_brdf = False
_C.MODEL_LIGHT.use_GT_light_envmap = False
# _C.MODEL_LIGHT.use_GT_light_axis = False
# _C.MODEL_LIGHT.use_GT_light_lamb = False
# _C.MODEL_LIGHT.use_GT_light_weight = False
_C.MODEL_LIGHT.load_GT_light_sg = False
_C.MODEL_LIGHT.use_GT_light_sg = False
_C.MODEL_LIGHT.load_pretrained_MODEL_BRDF = False
_C.MODEL_LIGHT.load_pretrained_MODEL_LIGHT = False
_C.MODEL_LIGHT.freeze_BRDF_Net = False
_C.MODEL_LIGHT.pretrained_pth_name = 'check_cascadeLight0_sg12_offset1.0/%s0_9.pth' # should not use for Rui's splits; this ckpt was trained with Zhengqin's CVPR'20 splits
_C.MODEL_LIGHT.use_scale_aware_loss = False
_C.MODEL_LIGHT.if_transform_to_LightNet_coords = False # if transform pred lighting to global LightNet coords
_C.MODEL_LIGHT.enable_list = 'axis_lamb_weight'
_C.MODEL_LIGHT.if_align_log_envmap = True # instead of align raw envmap_pred and envmap_gt
_C.MODEL_LIGHT.if_align_rerendering_envmap = False
_C.MODEL_LIGHT.if_clamp_coeff = True

_C.MODEL_LIGHT.depth_thres = 50.

_C.MODEL_LIGHT.DPT_baseline = CN()
_C.MODEL_LIGHT.DPT_baseline.enable = False
_C.MODEL_LIGHT.DPT_baseline.model = 'dpt_hybrid'
# _C.MODEL_LIGHT.DPT_baseline.if_freeze_backbone = False
_C.MODEL_LIGHT.DPT_baseline.if_share_patchembed = False
_C.MODEL_LIGHT.DPT_baseline.if_share_pretrained = False
_C.MODEL_LIGHT.DPT_baseline.if_batch_norm = False # in DPT output head
_C.MODEL_LIGHT.DPT_baseline.if_batch_norm_weight_override = False # in DPT output head
# _C.MODEL_LIGHT.DPT_baseline.if_group_norm = False # in DPT output head
_C.MODEL_LIGHT.DPT_baseline.if_imagenet_backbone = True
_C.MODEL_LIGHT.DPT_baseline.readout = 'ignore'
_C.MODEL_LIGHT.DPT_baseline.if_pos_embed = False
_C.MODEL_LIGHT.DPT_baseline.if_checkpoint = False
_C.MODEL_LIGHT.DPT_baseline.patch_size = 16
_C.MODEL_LIGHT.DPT_baseline.in_channels = 11

_C.MODEL_LIGHT.DPT_baseline.enable_as_single_est = False
_C.MODEL_LIGHT.DPT_baseline.enable_as_single_est_modality = 'weight'
_C.MODEL_LIGHT.DPT_baseline.enable_as_single_est_freeze_LightNet = False

_C.MODEL_LIGHT.DPT_baseline.use_vit_only = False

_C.MODEL_LIGHT.DPT_baseline.if_share_decoder_over_heads = True # [always True] used in constructing MODEL_ALL (ViT); currently treating each modality (e.g. albedo, roughness) as one decoder with one head

# _C.MODEL_LIGHT.DPT_baseline.dpt_hybrid = CN() # share with MODEL_BRDF

_C.MODEL_LIGHT.DPT_baseline.swin = CN()
_C.MODEL_LIGHT.DPT_baseline.swin.patch_size = 4

# _C.MODEL_LIGHT.pretrained_pth_name = ''

# ===== layout, objects, emitter
_C.MODEL_LAYOUT_EMITTER = CN()
_C.MODEL_LAYOUT_EMITTER.enable = False # enable model / modules
_C.MODEL_LAYOUT_EMITTER.enable_list = '' # enable model / modules
_C.MODEL_LAYOUT_EMITTER.enable_list_allowed = ['lo', 'ob', 'em', 'mesh', 'joint']
_C.MODEL_LAYOUT_EMITTER.loss_list = ''
_C.MODEL_LAYOUT_EMITTER.use_depth_as_input = False

_C.MODEL_LAYOUT_EMITTER.data = CN()
_C.MODEL_LAYOUT_EMITTER.data.OR = 'OR45'
_C.MODEL_LAYOUT_EMITTER.data.version = 'V4full'

_C.MODEL_LAYOUT_EMITTER.emitter = CN()
_C.MODEL_LAYOUT_EMITTER.emitter.if_freeze = False
_C.MODEL_LAYOUT_EMITTER.emitter.if_use_est_layout = False
_C.MODEL_LAYOUT_EMITTER.emitter.if_differentiable_layout_input = False
_C.MODEL_LAYOUT_EMITTER.emitter.if_train_with_reindexed_layout = False
_C.MODEL_LAYOUT_EMITTER.emitter.grid_size = 8
_C.MODEL_LAYOUT_EMITTER.emitter.est_type = 'cell_info'
_C.MODEL_LAYOUT_EMITTER.emitter.representation_type = '3SG' # 0ambient, 1ambient, 2ambient
_C.MODEL_LAYOUT_EMITTER.emitter.loss_type = 'L2' # [L2, KL]
_C.MODEL_LAYOUT_EMITTER.emitter.sigmoid = False
_C.MODEL_LAYOUT_EMITTER.emitter.softmax = False
_C.MODEL_LAYOUT_EMITTER.emitter.relative_dir = True
_C.MODEL_LAYOUT_EMITTER.emitter.scale_invariant_loss_for_cell_axis = True
_C.MODEL_LAYOUT_EMITTER.emitter.cls_agnostric = False
_C.MODEL_LAYOUT_EMITTER.emitter.loss = CN()
_C.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_axis_global = 4.
_C.MODEL_LAYOUT_EMITTER.emitter.loss.weight_light_ratio = 10.
_C.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_cls = 10.
_C.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_intensity = 0.2
_C.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_lamb = 0.3

_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net = CN() # better model than the vanilla model
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.enable = False # enable spatial-encoding network from per-pixel lighting, instead of image encoder-decoder
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.version = 'V2'
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_GT_light = True # use GT per-pixel lighting instead of predicting using LIGHT_NET
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_GT_brdf = True # use GT brdf instead of predicting using BRDF_NET
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.freeze_lightnet = True # freeze LIGHT_NET when using predictiion from LIGHT_NET
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.freeze_brdfnet = True # freeze LIGHT_NET when using predictiion from LIGHT_NET
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_weighted_axis = True
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.envHeight = 8
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.envWidth = 16

_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.sample_envmap = False
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_envmap_as_input = False

_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_img_feats_as_input = False
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.sample_BRDF_feats_instead_of_learn_feats = False
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.img_feats_channels = 8 # = 64 + 128 + 256 + 256 if sample_BRDF_feats_instead_of_learn_feats

_C.MODEL_LAYOUT_EMITTER.layout = CN()
_C.MODEL_LAYOUT_EMITTER.layout.if_freeze = False
_C.MODEL_LAYOUT_EMITTER.layout.loss = CN()
_C.MODEL_LAYOUT_EMITTER.layout.loss.cls_reg_ratio = 10
_C.MODEL_LAYOUT_EMITTER.layout.loss.obj_cam_ratio = 1
_C.MODEL_LAYOUT_EMITTER.layout.loss.weight_all = 1
_C.MODEL_LAYOUT_EMITTER.layout.if_train_with_reindexed = False
_C.MODEL_LAYOUT_EMITTER.layout.if_indept_encoder = True
# _C.MODEL_LAYOUT_EMITTER.layout.if_fully_differentiable = False # get rid of argmax in layout est -> bbox; not implememted yet
_C.MODEL_LAYOUT_EMITTER.layout.if_estcls_in_loss = False
# _C.MODEL_LAYOUT_EMITTER.layout.if_argmax_in_results = True

_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline = CN()
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.enable = False
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_pos_embed = True
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.model = 'dpt_hybrid'
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.readout = 'project'
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_imagenet_backbone = True
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.patch_size = 16
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_share_encoder_over_modalities = True
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_share_decoder_over_heads = False # e.g. for layout estimation there are two heads: cam, lo. Set to True to use indept encoder for each
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_indept_MLP_heads = False
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.if_indept_MLP_heads_if_layer_norm = False
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.ViT_pool = 'mean'

_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.dpt_hybrid = CN()
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.dpt_hybrid.N_layers_encoder = 6
_C.MODEL_LAYOUT_EMITTER.layout.ViT_baseline.dpt_hybrid.N_layers_decoder = 6

_C.MODEL_LAYOUT_EMITTER.mesh = CN()
_C.MODEL_LAYOUT_EMITTER.mesh.tmn_subnetworks = 2
_C.MODEL_LAYOUT_EMITTER.mesh.face_samples = 1
_C.MODEL_LAYOUT_EMITTER.mesh.with_edge_classifier = True
_C.MODEL_LAYOUT_EMITTER.mesh.neighbors = 30
_C.MODEL_LAYOUT_EMITTER.mesh.loss = 'SVRLoss' # ['SVRLoss', 'ReconLoss']
_C.MODEL_LAYOUT_EMITTER.mesh.original_path = ''
_C.MODEL_LAYOUT_EMITTER.mesh.original_path_local = '/newfoundland2/ruizhu/siggraphasia20dataset/uv_mapped'
_C.MODEL_LAYOUT_EMITTER.mesh.original_path_cluster = ['/siggraphasia20dataset/uv_mapped', '', '']
_C.MODEL_LAYOUT_EMITTER.mesh.sampled_path = ''
_C.MODEL_LAYOUT_EMITTER.mesh.sampled_path_local = '/home/ruizhu/Documents/data/OR-sampledMeshes'
_C.MODEL_LAYOUT_EMITTER.mesh.sampled_path_cluster = ['/ruidata/OR-sampledMeshes', '', '']
_C.MODEL_LAYOUT_EMITTER.mesh.if_use_vtk = True

_C.MODEL_LAYOUT_EMITTER.mesh_obj = CN()
_C.MODEL_LAYOUT_EMITTER.mesh_obj.log_valid_objs = False
# filter invalid frames with 0 valid objects, and filter invalid objects in dataloader
_C.MODEL_LAYOUT_EMITTER.mesh_obj.if_clip_boxes_train = True # randomly sample N objects if there are too many
_C.MODEL_LAYOUT_EMITTER.mesh_obj.clip_boxes_train_to = 5
_C.MODEL_LAYOUT_EMITTER.mesh_obj.if_use_only_valid_objs = True
_C.MODEL_LAYOUT_EMITTER.mesh_obj.valid_bbox_vis_ratio = 0.1

_C.MODEL_LAYOUT_EMITTER.mesh_obj.if_pre_filter_invalid_frames = False # using e.g./home/ruizhu/Documents/Projects/semanticInverse/train/data/openrooms/list_ORmini-val/list/obj_list.pickle
_C.MODEL_LAYOUT_EMITTER.mesh_obj.if_skip_invalid_frames = True # skip invalid frames in dataloader.__getitem()__

# ===== detectron
_C.MODEL_DETECTRON = CN()
_C.MODEL_DETECTRON.enable = False
_C.MODEL_DETECTRON.pretrained = True
_C.MODEL_DETECTRON.thresh = 0.75
_C.MODEL_DETECTRON.nms_thresh = 0.6

# ===== material cls (Yu-Ying)
_C.MODEL_MATCLS = CN()
_C.MODEL_MATCLS.enable = False
_C.MODEL_MATCLS.num_classes = 886
_C.MODEL_MATCLS.num_classes_sup = 9 # (+ 1 for unlabelled)
_C.MODEL_MATCLS.if_est_sup = False # substance loss
_C.MODEL_MATCLS.loss_sup_weight = 0.5
_C.MODEL_MATCLS.real_images_list = '/newfoundland2/ruizhu/yyyeh/OpenRoomScanNetView/real_images_list.txt'

# ===== material segmentation
_C.MODEL_MATSEG = CN()
_C.MODEL_MATSEG.enable = False
_C.MODEL_MATSEG.arch = 'resnet101'
_C.MODEL_MATSEG.pretrained = True
_C.MODEL_MATSEG.if_freeze = False
_C.MODEL_MATSEG.fix_bn = False
_C.MODEL_MATSEG.embed_dims = 4
_C.MODEL_MATSEG.if_guide = False
_C.MODEL_MATSEG.guide_channels = 32
_C.MODEL_MATSEG.use_as_input = False
_C.MODEL_MATSEG.use_semseg_as_input = False
_C.MODEL_MATSEG.load_pretrained_pth = False
_C.MODEL_MATSEG.pretrained_pth = ''
_C.MODEL_MATSEG.use_pred_as_input = False
_C.MODEL_MATSEG.if_save_embedding = False

_C.MODEL_MATSEG.albedo_pooling_debug = False

_C.MODEL_MATSEG.if_albedo_pooling = False
_C.MODEL_MATSEG.albedo_pooling_from = 'gt' # ['gt', 'pred']

_C.MODEL_MATSEG.if_albedo_asso_pool_conv = False

_C.MODEL_MATSEG.if_albedo_pac_pool = False
_C.MODEL_MATSEG.if_albedo_pac_pool_debug_deform = False
_C.MODEL_MATSEG.if_albedo_pac_pool_keep_input = True
_C.MODEL_MATSEG.if_albedo_DatasetNew_test_pool_mean = False # True: return mean of pooled tensors; False: return stacked
_C.MODEL_MATSEG.albedo_pac_pool_mean_layers = 'xin6'
_C.MODEL_MATSEG.albedo_pac_pool_mean_layers_allowed = 'x6_xin1_xin2_xin3_xin4_xin5_xin6'
# _C.MODEL_MATSEG.albedo_pac_pool_deform_layers = 'xin6'
# _C.MODEL_MATSEG.albedo_pac_pool_deform_layers_allowed = 'x6_xin1_xin2_xin3_xin4_xin5_xin6'

_C.MODEL_MATSEG.if_albedo_pac_conv = False
_C.MODEL_MATSEG.if_albedo_pac_conv_keep_input = True
_C.MODEL_MATSEG.if_albedo_pac_conv_mean = False # True: return mean of pooled tensors; False: return stacked
_C.MODEL_MATSEG.if_albedo_pac_conv_normalize_kernel = True
_C.MODEL_MATSEG.if_albedo_pac_conv_DCN = False
_C.MODEL_MATSEG.albedo_pac_conv_deform_layers = 'xin6'
_C.MODEL_MATSEG.albedo_pac_conv_deform_layers_allowed = 'x6_xin1_xin2_xin3_xin4_xin5_xin6'

_C.MODEL_MATSEG.albedo_pac_conv_mean_layers = 'xin6'
_C.MODEL_MATSEG.albedo_pac_conv_mean_layers_allowed = 'x6_xin1_xin2_xin3_xin4_xin5_xin6'

_C.MODEL_MATSEG.if_albedo_safenet = False
_C.MODEL_MATSEG.if_albedo_safenet_keep_input = True
_C.MODEL_MATSEG.if_albedo_safenet_normalize_embedding = False
_C.MODEL_MATSEG.if_albedo_safenet_use_pacnet_affinity = False
# _C.MODEL_MATSEG.if_albedo_safenet_mean = False # True: return mean of pooled tensors; False: return stacked
_C.MODEL_MATSEG.albedo_safenet_affinity_layers = 'xin3'
_C.MODEL_MATSEG.albedo_safenet_affinity_layers_allowed = 'x6_xin1_xin2_xin3_xin4_xin5_xin6'

# ===== semantic segmentation

_C.MODEL_SEMSEG = CN()
_C.MODEL_SEMSEG.enable = False
_C.MODEL_SEMSEG.use_as_input = False
_C.MODEL_SEMSEG.wallseg_only = False
_C.MODEL_SEMSEG.semseg_path = ''
_C.MODEL_SEMSEG.semseg_path_local = '/home/ruizhu/Documents/Projects/semseg'
_C.MODEL_SEMSEG.semseg_path_cluster = ['/viscompfs/users/ruizhu/semseg/', '', '']
_C.MODEL_SEMSEG.config_file = 'configs/openrooms/openrooms_pspnet50.yaml'
# _C.MODEL_SEMSEG.semseg_colors = 'data/openrooms/openrooms_colors.txt'
_C.MODEL_SEMSEG.if_freeze = False
_C.MODEL_SEMSEG.fix_bn = False
_C.MODEL_SEMSEG.if_guide = False
_C.MODEL_SEMSEG.load_pretrained_pth = False
_C.MODEL_SEMSEG.pretrained_pth = 'exp/openrooms/pspnet50V3_2gpu_100k/model/train_epoch_23_tid_147000.pth'
_C.MODEL_SEMSEG.semseg_ignore_label = 0
_C.MODEL_SEMSEG.semseg_classes = 46
_C.MODEL_SEMSEG.pspnet_version = 50 # [50, 101]

# ===== everything model
_C.MODEL_ALL = CN()
_C.MODEL_ALL.enable = False # enable model / modules
_C.MODEL_ALL.enable_list = '' # enable model / modules
_C.MODEL_ALL.enable_list_allowed = ['al', 'no', 'de', 'ro', 'lo', 'li']
_C.MODEL_ALL.loss_list = ''

_C.MODEL_ALL.ViT_baseline = CN()
_C.MODEL_ALL.ViT_baseline.enable = True # enable model / modules
# _C.MODEL_ALL.ViT_baseline.use_vit_only = False # only works for hybrid (which is the case here)
# _C.MODEL_ALL.ViT_baseline.if_pos_embed = True
# _C.MODEL_ALL.ViT_baseline.readout = 'project'
# _C.MODEL_ALL.ViT_baseline.if_imagenet_backbone = True
# _C.MODEL_ALL.ViT_baseline.patch_size = 16
_C.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage0 = True # brdf + lo
_C.MODEL_ALL.ViT_baseline.if_share_encoder_over_modalities_stage1 = True # lightnet
_C.MODEL_ALL.ViT_baseline.if_share_decoder_over_BRDF_modalities = False
_C.MODEL_ALL.ViT_baseline.if_share_pretrained_over_BRDF_modalities = False
# _C.MODEL_ALL.ViT_baseline.if_share_decoder_over_heads = False # e.g. for layout estimation there are two heads: cam, lo. Set to True to use indept encoder for each
_C.MODEL_ALL.ViT_baseline.if_indept_MLP_heads = False
_C.MODEL_ALL.ViT_baseline.if_indept_MLP_heads_if_layer_norm = False
_C.MODEL_ALL.ViT_baseline.ViT_pool = 'mean'
_C.MODEL_ALL.ViT_baseline.N_layers_encoder_stage0 = 4
_C.MODEL_ALL.ViT_baseline.N_layers_decoder_stage0 = 4
_C.MODEL_ALL.ViT_baseline.N_layers_encoder_stage1 = 4
_C.MODEL_ALL.ViT_baseline.N_layers_decoder_stage1 = 4

_C.MODEL_ALL.ViT_baseline.if_UNet_lighting = False

# _C.MODEL_ALL.ViT_baseline.depth = CN()
# _C.MODEL_ALL.ViT_baseline.depth.activation = 'tanh'


# ===== solver

_C.SOLVER = CN()
_C.SOLVER.method = 'adam'
_C.SOLVER.lr = 3e-5
_C.SOLVER.if_warm_up = False
_C.SOLVER.weight_decay = 0.00001
# _C.SOLVER.max_iter = 10000000
_C.SOLVER.max_epoch = 1000
_C.SOLVER.ims_per_batch = 16
_C.SOLVER.if_test_dataloader = False


_C.TRAINING = CN()
_C.TRAINING.MAX_CKPT_KEEP = 5

_C.TEST = CN()
_C.TEST.ims_per_batch = 16
_C.TEST.vis_max_samples = 20

_C.seed = 123
# _C.num_gpus = 1
# _C.num_epochs = 100
_C.resume_dir = None
_C.print_interval = 10
_C.flush_secs = 10