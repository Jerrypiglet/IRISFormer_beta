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

_C.model = CN()
_C.model.DTYPE = "float32"
_C.model.arch = 'resnet101'
_C.model.pretrained = True
_C.model.embed_dims = 2
_C.model.fix_bn = False


_C.dataset = CN()
_C.dataset.root_dir = '/new_disk2/yuzh/PlaneNetData/'
# _C.dataset.batch_size = 16
_C.dataset.num_workers = 8
_C.dataset.if_val_dist = True
_C.dataset.if_hdr = False

_C.DATA = CN()
_C.DATA.IM_HEIGHT = 192
_C.DATA.IM_WIDTH = 256


_C.solver = CN()
_C.solver.method = 'adam'
_C.solver.lr = 0.0001
_C.solver.weight_decay = 0.00001
_C.solver.max_iter = 10000000
_C.solver.max_epoch = 1000
_C.solver.ims_per_batch = 16

_C.TRAINING = CN()
_C.TRAINING.MAX_CKPT_KEEP = 10

_C.test = CN()
_C.test.ims_per_batch = 16

_C.seed = 123
# _C.num_gpus = 1
_C.num_epochs = 100
_C.resume_dir = None
_C.print_interval = 10