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

_C.PATH = CN()
_C.PATH.root = ''
_C.PATH.root_local = '/home/ruizhu/Documents/Projects/semanticInverse/train'
_C.PATH.root_cluster = '.'
_C.PATH.semseg_colors_path = 'data/openrooms/openrooms_colors.txt'
_C.PATH.semseg_names_path = 'data/openrooms/openrooms_names.txt'
_C.PATH.total3D_colors_path = 'data/openrooms/total3D_colors'

_C.PATH.total3D_lists_path = 'data/openrooms/list_OR_V4full'
_C.PATH.total3D_lists_path_if_zhengqinCVPR = False
_C.PATH.total3D_lists_path_zhengqinCVPR = 'data/openrooms/list_OR_V4full_zhengqinCVPR'

_C.PATH.matcls_matIdG1_path = 'data/openrooms/matIdGlobal1.txt'
_C.PATH.matcls_matIdG2_path = 'data/openrooms/matIdGlobal2.txt'
_C.PATH.dcn_path = ''
_C.PATH.dcn_cluster = '/viscompfs/users/ruizhu/lib/Deformable-Convolution-V2-PyTorch'
_C.PATH.dcn_local = '/home/ruizhu/Documents/Projects/semanticInverse/third-parties_outside/Deformable-Convolution-V2-PyTorch'
_C.PATH.torch_home_path = ''
_C.PATH.torch_home_local = '/home/ruizhu/Documents/Projects/semanticInverse/'
_C.PATH.torch_home_cluster = '/viscompfs/users/ruizhu/'
# _C.DATA.semseg_colors_path = 'data/openrooms/openrooms_colors.txt'
# _C.DATA.semseg_names_path = 'data/openrooms/openrooms_names.txt'
_C.PATH.OR4X_mapping_catInt_to_RGB = ['data/openrooms/total3D_colors/OR4X_mapping_catInt_to_RGB_light.pkl', 'data/openrooms/total3D_colors/OR4X_mapping_catInt_to_RGB_dark.pkl']
_C.PATH.OR4X_mapping_catStr_to_RGB = ['data/openrooms/total3D_colors/OR4X_mapping_catStr_to_RGB_light.pkl', 'data/openrooms/total3D_colors/OR4X_mapping_catStr_to_RGB_dark.pkl']

# ===== dataset

_C.DATASET = CN()
_C.DATASET.mini = False # load mini OR from SSD to enable faster dataloading for debugging purposes etc.
_C.DATASET.dataset_name = 'openrooms'
_C.DATASET.dataset_path = ''
_C.DATASET.dataset_path_local = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms'
_C.DATASET.dataset_path_cluster = '/siggraphasia20dataset/code/Routine/DatasetCreation/'
_C.DATASET.dataset_path_test = ''
_C.DATASET.dataset_path_test_local = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_test'
_C.DATASET.dataset_path_test_cluster = '/eccv20dataset/DatasetNew_test'
_C.DATASET.png_path = ''
_C.DATASET.png_path_local = '/data/ruizhu/OR-pngs'
_C.DATASET.png_path_cluster = '/siggraphasia20dataset/pngs'

_C.DATASET.layout_emitter_path = ''
# _C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-detachEmitter-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-detachEmitterRERE-OR45_total3D_train_test_data'
_C.DATASET.layout_emitter_path_local = '/data/ruizhu/OR-V4full-detachEmitterRERERE20210425-OR45_total3D_train_test_data'

# _C.DATASET.layout_emitter_path_cluster = '/ruidata/OR-V4full-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_cluster = '/ruidata/OR-V4full-detachEmitter-OR45_total3D_train_test_data'
# _C.DATASET.layout_emitter_path_cluster = '/ruidata/OR-V4full-detachEmitterRERE-OR45_total3D_train_test_data'
_C.DATASET.layout_emitter_path_cluster = '/ruidata/OR-V4full-detachEmitterRERERE20210425-OR45_total3D_train_test_data_20210430'

_C.DATASET.envmap_path = ''
_C.DATASET.envmap_path_local = '/home/ruizhu/Documents/data/EnvDataset/'
_C.DATASET.envmap_path_cluster = '/siggraphasia20dataset/EnvDataset/'

_C.DATASET.matpart_path = ''
_C.DATASET.matpart_path_local = '/data/ruizhu/OR-matpart'
_C.DATASET.matpart_path_cluster = '/siggraphasia20dataset/code/Routine/DatasetCreation/'
_C.DATASET.matori_path = ''
_C.DATASET.matori_path_local = '/newfoundland2/ruizhu/siggraphasia20dataset/BRDFOriginDataset/'
_C.DATASET.matori_path_cluster = '/siggraphasia20dataset/BRDFOriginDataset/'

# _C.DATASET.dataset_list = 'data/openrooms/list_OR_V4full/list'
_C.DATASET.dataset_list = ''
# _C.DATASET.dataset_path_mini = '/data/ruizhu/openrooms_mini'
# _C.DATASET.dataset_list_mini = 'data/openrooms/list_ORmini/list'
_C.DATASET.dataset_path_mini = '/data/ruizhu/openrooms_mini-val'
_C.DATASET.dataset_list_mini = 'data/openrooms/list_ORmini-val/list'

_C.DATASET.num_workers = 8
_C.DATASET.if_val_dist = True

# ===== data loading configs

_C.DATA = CN()
_C.DATA.if_load_png_not_hdr = False # load png as input image instead of hdr image
_C.DATA.if_augment_train = False
_C.DATA.im_height = 240
_C.DATA.im_width = 320
_C.DATA.load_semseg_gt = False
_C.DATA.load_matseg_gt = False
_C.DATA.load_brdf_gt = False
_C.DATA.load_light_gt = False
_C.DATA.load_layout_emitter_gt = False
_C.DATA.data_read_list = ''
_C.DATA.data_read_list_allowed = ['al', 'no', 'de', 'ro', \
    'lo', 'em']
_C.DATA.load_matcls_gt = False

# ===== BRDF
_C.MODEL_BRDF = CN()
_C.MODEL_BRDF.enable = False
_C.MODEL_BRDF.if_freeze = False
# _C.MODEL_BRDF.enable_list = ['al', 'no', 'de', 'ro', 'li']
_C.MODEL_BRDF.enable_list = '' # `al_no_de_ro`
_C.MODEL_BRDF.enable_list_allowed = ['al', 'no', 'de', 'ro']
_C.MODEL_BRDF.load_pretrained_pth = False
_C.MODEL_BRDF.loss_list = ''
_C.MODEL_BRDF.albedoWeight = 1.5
_C.MODEL_BRDF.normalWeight = 1.0
_C.MODEL_BRDF.roughWeight = 0.5
_C.MODEL_BRDF.depthWeight = 0.5
_C.MODEL_BRDF.if_debug_arch = False
_C.MODEL_BRDF.enable_BRDF_decoders = False
# _C.MODEL_BRDF.is_all_light = True
_C.MODEL_BRDF.enable_semseg_decoder = False
_C.MODEL_BRDF.semseg_PPM = True
_C.MODEL_BRDF.pretrained_pth_name = 'check_cascade0_w320_h240/%s0_13.pth' # should not use for Rui's splits; this ckpt was trained with Zhengqin's CVPR'20 splits
# _C.MODEL_BRDF.pretrained_pth_name = ''
_C.MODEL_BRDF.encoder_exclude = '' # e.g. 'x4_x5
_C.MODEL_BRDF.use_scale_aware_depth = False
_C.MODEL_BRDF.depth_activation = 'sigmoid'
_C.MODEL_BRDF.use_scale_aware_albedo = False

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
_C.MODEL_LIGHT.use_GT_light = False
_C.MODEL_LIGHT.load_pretrained_MODEL_BRDF = False
_C.MODEL_LIGHT.load_pretrained_MODEL_LIGHT = False
_C.MODEL_LIGHT.freeze_BRDF_Net = True
_C.MODEL_LIGHT.pretrained_pth_name = 'check_cascadeLight0_sg12_offset1.0/%s0_9.pth' # should not use for Rui's splits; this ckpt was trained with Zhengqin's CVPR'20 splits
_C.MODEL_LIGHT.use_scale_aware_loss = False

# _C.MODEL_LIGHT.pretrained_pth_name = ''

# ===== layout, emitter
_C.MODEL_LAYOUT_EMITTER = CN()
_C.MODEL_LAYOUT_EMITTER.enable = False # enable model / modules
_C.MODEL_LAYOUT_EMITTER.enable_list = 'lo_em' # enable model / modules
_C.MODEL_LAYOUT_EMITTER.enable_list_allowed = ['lo', 'em']
_C.MODEL_LAYOUT_EMITTER.loss_list = ''
_C.MODEL_LAYOUT_EMITTER.use_depth_as_input = False

_C.MODEL_LAYOUT_EMITTER.data = CN()
_C.MODEL_LAYOUT_EMITTER.data.OR = 'OR45'
_C.MODEL_LAYOUT_EMITTER.data.version = 'V4full'

_C.MODEL_LAYOUT_EMITTER.emitter = CN()
_C.MODEL_LAYOUT_EMITTER.emitter.grid_size = 8
_C.MODEL_LAYOUT_EMITTER.emitter.est_type = 'cell_info'
_C.MODEL_LAYOUT_EMITTER.emitter.representation_type = '0ambient' # 0ambient, 1ambient, 2ambient
_C.MODEL_LAYOUT_EMITTER.emitter.relative_dir = True
_C.MODEL_LAYOUT_EMITTER.emitter.loss_type = 'L2' # [L2, KL]
_C.MODEL_LAYOUT_EMITTER.emitter.scale_invariant_loss_for_cell_axis = True
_C.MODEL_LAYOUT_EMITTER.emitter.sigmoid = False
_C.MODEL_LAYOUT_EMITTER.emitter.softmax = False
_C.MODEL_LAYOUT_EMITTER.emitter.cls_agnostric = False
_C.MODEL_LAYOUT_EMITTER.emitter.loss = CN()
_C.MODEL_LAYOUT_EMITTER.emitter.loss.weight_cell_axis_global = 4.
_C.MODEL_LAYOUT_EMITTER.emitter.loss.weight_light_ratio = 100.
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

_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.sample_envmap = False
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_envmap_as_input = False

_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_img_feats_as_input = False
_C.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.img_feats_channels = 8

_C.MODEL_LAYOUT_EMITTER.layout = CN()
_C.MODEL_LAYOUT_EMITTER.layout.loss = CN()
_C.MODEL_LAYOUT_EMITTER.layout.loss.cls_reg_ratio = 10
_C.MODEL_LAYOUT_EMITTER.layout.loss.obj_cam_ratio = 1

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
_C.MODEL_MATSEG.embed_dims = 2
_C.MODEL_MATSEG.if_guide = False
_C.MODEL_MATSEG.guide_channels = 32
_C.MODEL_MATSEG.use_as_input = False
_C.MODEL_MATSEG.use_semseg_as_input = False
_C.MODEL_MATSEG.load_pretrained_pth = False
_C.MODEL_MATSEG.pretrained_pth = ''
_C.MODEL_MATSEG.use_pred_as_input = False

_C.MODEL_MATSEG.albedo_pooling_debug = False

_C.MODEL_MATSEG.if_albedo_pooling = False
_C.MODEL_MATSEG.albedo_pooling_from = 'gt' # ['gt', 'pred']

_C.MODEL_MATSEG.if_albedo_asso_pool_conv = False

_C.MODEL_MATSEG.if_albedo_pac_pool = False
_C.MODEL_MATSEG.if_albedo_pac_pool_debug_deform = False
_C.MODEL_MATSEG.if_albedo_pac_pool_keep_input = True
_C.MODEL_MATSEG.if_albedo_pac_pool_mean = False # True: return mean of pooled tensors; False: return stacked
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
_C.MODEL_SEMSEG.semseg_path_cluster = '/viscompfs/users/ruizhu/semseg/'
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

# ===== solver

_C.SOLVER = CN()
_C.SOLVER.method = 'adam'
_C.SOLVER.lr = 0.0001
_C.SOLVER.weight_decay = 0.00001
_C.SOLVER.max_iter = 10000000
_C.SOLVER.max_epoch = 1000
_C.SOLVER.ims_per_batch = 16
_C.SOLVER.if_test_dataloader = False


_C.TRAINING = CN()
_C.TRAINING.MAX_CKPT_KEEP = 5

_C.TEST = CN()
_C.TEST.ims_per_batch = 16
_C.TEST.vis_max_samples = 50

_C.seed = 123
# _C.num_gpus = 1
# _C.num_epochs = 100
_C.resume_dir = None
_C.print_interval = 10