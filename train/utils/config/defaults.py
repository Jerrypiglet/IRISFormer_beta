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

_C.MODEL_SEG = CN()
_C.MODEL_SEG.enable = True
_C.MODEL_SEG.arch = 'resnet101'
_C.MODEL_SEG.pretrained = True
_C.MODEL_SEG.if_freeze = False
_C.MODEL_SEG.fix_bn = False
_C.MODEL_SEG.embed_dims = 2
_C.MODEL_SEG.if_guide = False
_C.MODEL_SEG.guide_channels = 32

_C.MODEL_BRDF = CN()
_C.MODEL_BRDF.enable = True
_C.MODEL_BRDF.load_pretrained_pth = False
_C.MODEL_BRDF.pretrained_pth_name = 'check_cascade0_w320_h240'
_C.MODEL_BRDF.albedoWeight = 1.5
_C.MODEL_BRDF.normalWeight = 1.0
_C.MODEL_BRDF.roughWeight = 0.5
_C.MODEL_BRDF.depthWeight = 0.5
_C.MODEL_BRDF.if_debug_arch = False

_C.MODEL_BRDF.enable_semseg_decoder = False
_C.MODEL_BRDF.semseg_ignore_label = 0
_C.MODEL_BRDF.semseg_classes = 44
_C.MODEL_BRDF.semseg_colors_path = 'data/openrooms_colors.txt'
_C.MODEL_BRDF.semseg_names_path = 'data/openrooms_names.txt'



_C.MODEL_SEMSEG = CN()
_C.MODEL_SEMSEG.enable = True
_C.MODEL_SEMSEG.config_file = 'configs/ade20k/ade20k_pspnet50.yaml'
_C.MODEL_SEMSEG.if_freeze = False
_C.MODEL_SEMSEG.fix_bn = False
_C.MODEL_SEMSEG.if_guide = False
_C.MODEL_SEMSEG.load_pretrained_pth = True
_C.MODEL_SEMSEG.pretrained_pth_name = ''


# _C.MODEL_SEMSEG.configs = ()

_C.DATASET = CN()
_C.DATASET.root_dir = '/new_disk2/yuzh/PlaneNetData/'
# _C.DATASET.batch_size = 16
_C.DATASET.num_workers = 8
_C.DATASET.if_val_dist = True
# _C.DATASET.if_hdr = False

_C.DATA = CN()
_C.DATA.im_height = 240
_C.DATA.im_width = 320


_C.SOLVER = CN()
_C.SOLVER.method = 'adam'
_C.SOLVER.lr = 0.0001
_C.SOLVER.weight_decay = 0.00001
_C.SOLVER.max_iter = 10000000
_C.SOLVER.max_epoch = 1000
_C.SOLVER.ims_per_batch = 16

_C.TRAINING = CN()
_C.TRAINING.MAX_CKPT_KEEP = 10

_C.TEST = CN()
_C.TEST.ims_per_batch = 16

_C.seed = 123
# _C.num_gpus = 1
# _C.num_epochs = 100
_C.resume_dir = None
_C.print_interval = 10