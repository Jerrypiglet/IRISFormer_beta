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
# _C.DATA.semseg_colors_path = 'data/openrooms/openrooms_colors.txt'
# _C.DATA.semseg_names_path = 'data/openrooms/openrooms_names.txt'

_C.MODEL_BRDF = CN()
_C.MODEL_BRDF.enable = True
# _C.MODEL_BRDF.enable_list = ['al', 'no', 'de', 'ro']
_C.MODEL_BRDF.enable_list = 'al_no_de_ro'
_C.MODEL_BRDF.load_pretrained_pth = False
_C.MODEL_BRDF.pretrained_pth_name = 'check_cascade0_w320_h240'
_C.MODEL_BRDF.albedoWeight = 1.5
_C.MODEL_BRDF.normalWeight = 1.0
_C.MODEL_BRDF.roughWeight = 0.5
_C.MODEL_BRDF.depthWeight = 0.5
_C.MODEL_BRDF.if_debug_arch = False
_C.MODEL_BRDF.enable_BRDF_decoders = True

_C.MODEL_BRDF.enable_semseg_decoder = False
_C.MODEL_BRDF.semseg_PPM = False


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
_C.MODEL_MATSEG.if_albedo_pac_pool_keep_input = True
_C.MODEL_MATSEG.if_albedo_pac_pool_mean = False # True: return mean of pooled tensors; False: return stacked
_C.MODEL_MATSEG.albedo_pac_pool_mean_layers = 'xin6'
_C.MODEL_MATSEG.albedo_pac_pool_mean_layers_allowed = 'x6_xin1_xin2_xin3_xin4_xin5_xin6'

_C.MODEL_MATSEG.if_albedo_pac_conv = False
_C.MODEL_MATSEG.if_albedo_pac_conv_keep_input = True
_C.MODEL_MATSEG.if_albedo_pac_conv_mean = False # True: return mean of pooled tensors; False: return stacked
_C.MODEL_MATSEG.if_albedo_pac_conv_normalize_kernel = True
_C.MODEL_MATSEG.albedo_pac_conv_mean_layers = 'xin6'
_C.MODEL_MATSEG.albedo_pac_conv_mean_layers_allowed = 'x6_xin1_xin2_xin3_xin4_xin5_xin6'

_C.MODEL_MATSEG.if_albedo_safenet = False
_C.MODEL_MATSEG.if_albedo_safenet_keep_input = True
_C.MODEL_MATSEG.if_albedo_safenet_normalize_embedding = False
_C.MODEL_MATSEG.if_albedo_safenet_use_pacnet_affinity = False
# _C.MODEL_MATSEG.if_albedo_safenet_mean = False # True: return mean of pooled tensors; False: return stacked
_C.MODEL_MATSEG.albedo_safenet_affinity_layers = 'xin3'
_C.MODEL_MATSEG.albedo_safenet_affinity_layers_allowed = 'x6_xin1_xin2_xin3_xin4_xin5_xin6'

_C.MODEL_SEMSEG = CN()
_C.MODEL_SEMSEG.enable = False
_C.MODEL_SEMSEG.use_as_input = False
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


# _C.MODEL_SEMSEG.configs = ()

_C.DATASET = CN()
_C.DATASET.dataset_name = 'openrooms'
_C.DATASET.dataset_path = ''
_C.DATASET.dataset_path_local = 'dataset/openrooms'
_C.DATASET.dataset_path_cluster = '/siggraphasia20dataset/code/Routine/DatasetCreation/'
_C.DATASET.dataset_list = 'data/openrooms/list'

# _C.DATASET.batch_size = 16
_C.DATASET.num_workers = 8
_C.DATASET.if_val_dist = True
# _C.DATASET.if_hdr = False

_C.DATA = CN()
_C.DATA.im_height = 240
_C.DATA.im_width = 320
_C.DATA.load_semseg_gt = False
_C.DATA.load_matseg_gt = False
_C.DATA.load_brdf_gt = True
_C.DATA.semseg_ignore_label = 0
_C.DATA.semseg_classes = 46

_C.SOLVER = CN()
_C.SOLVER.method = 'adam'
_C.SOLVER.lr = 0.0001
_C.SOLVER.weight_decay = 0.00001
_C.SOLVER.max_iter = 10000000
_C.SOLVER.max_epoch = 1000
_C.SOLVER.ims_per_batch = 16

_C.TRAINING = CN()
_C.TRAINING.MAX_CKPT_KEEP = 20

_C.TEST = CN()
_C.TEST.ims_per_batch = 16

_C.seed = 123
# _C.num_gpus = 1
# _C.num_epochs = 100
_C.resume_dir = None
_C.print_interval = 10