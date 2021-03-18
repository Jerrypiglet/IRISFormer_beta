import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
from tqdm import tqdm
import time
import os, sys, inspect
pwdpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from pathlib import Path
os.system('touch %s/models_def/__init__.py'%pwdpath)
os.system('touch %s/utils/__init__.py'%pwdpath)
os.system('touch %s/__init__.py'%pwdpath)
print('started.' + pwdpath)
PACNET_PATH = Path(pwdpath) / 'third-parties' / 'pacnet'
sys.path.insert(0, str(PACNET_PATH))
print(sys.path)

from dataset_openroomsV4_total3d_matcls_real import openrooms, collate_fn_OR
from train_funcs_joint_all import get_labels_dict_joint, val_epoch_joint, vis_val_epoch_joint, forward_joint, get_time_meters_joint

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.config import cfg
from utils.bin_mean_shift import Bin_Mean_Shift
from utils.bin_mean_shift_3 import Bin_Mean_Shift_3
from utils.bin_mean_shift_N import Bin_Mean_Shift_N
from utils.comm import synchronize
from utils.utils_misc import *
from utils.utils_dataloader import make_data_loader
from utils.utils_training import reduce_loss_dict, check_save, print_gpu_usage, time_meters_to_string, find_free_port
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR

import utils.utils_config as utils_config
from utils.utils_envs import set_up_envs

from utils.utils_vis import vis_index_map

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--data_root', default=None, help='path to input images')
parser.add_argument('--task_name', type=str, default='tmp', help='task name (e.g. N1: disp ref)')
parser.add_argument('--task_split', type=str, default='train', help='train, val, test', choices={"train", "val", "test"})
# Fine tune the model
parser.add_argument('--isFineTune', action='store_true', help='fine-tune the model')
parser.add_argument("--if_val", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_vis", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--epochIdFineTune', type=int, default = 0, help='the training of epoch of the loaded model')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.5, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.5, help='the weight for depth component')   
parser.add_argument('--reconstWeight', type=float, default=10, help='the weight for reconstruction error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for the rendering' )
# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')

# Rui
# Device
parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument("--master_port", type=str, default='8914')

# DEBUG
parser.add_argument('--debug', action='store_true', help='Debug eval')

parser.add_argument('--ifMatMapInput', action='store_true', help='using mask as additional input')
parser.add_argument('--ifDataloaderOnly', action='store_true', help='benchmark dataloading overhead')
parser.add_argument('--if_cluster', action='store_true', help='if using cluster')
parser.add_argument('--if_hdr_input_matseg', action='store_true', help='if using hdr images')
parser.add_argument('--eval_every_iter', type=int, default=2000, help='')
parser.add_argument('--save_every_iter', type=int, default=2000, help='')
parser.add_argument('--debug_every_iter', type=int, default=2000, help='')
parser.add_argument('--max_iter', type=int, default=-1, help='')
parser.add_argument('--invalid_index', type=int, default = 0, help='index for invalid aread (e.g. windows, lights)')

# Pre-training
parser.add_argument('--resume', type=str, help='resume training; can be full path (e.g. tmp/checkpoint0.pth.tar) or taskname (e.g. tmp); [to continue the current task, use: resume]', default='NoCkpt')
parser.add_argument('--reset_latest_ckpt', action='store_true', help='remove latest_checkpoint file')
parser.add_argument('--reset_scheduler', action='store_true', help='')
parser.add_argument('--reset_lr', action='store_true', help='')
# debug
parser.add_argument("--mini_val", type=str2bool, nargs='?', const=True, default=False)
# to get rid of
parser.add_argument('--test_real', action='store_true', help='')


parser.add_argument('--replaced_keys', nargs='+', help='Replace those keys in the model', required=False)
parser.add_argument('--replacedby', nargs='+', help='... with those keys from ckpt. Must be in the same length as ``replace_leys``', required=False)

parser.add_argument(
    "--config-file",
    default=os.path.join(pwdpath, "configs/config.yaml"),
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "params",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

# The detail model setting
opt = parser.parse_args()
# print(opt)
# os.environ['MASETER_PORT'] = str(nextPort(int(opt.master_port)))
os.environ['MASETER_PORT'] = str(find_free_port())
cfg.merge_from_file(opt.config_file)
# cfg.merge_from_list(opt.params)
cfg = utils_config.merge_cfg_from_list(cfg, opt.params)
opt.cfg = cfg

# >>>>>>>>>>>>> A bunch of modularised set-ups
# opt.gpuId = opt.deviceIds[0]
from utils.utils_envs import set_up_dist
handle = set_up_dist(opt)

from utils.utils_envs import set_up_folders
set_up_folders(opt)

from utils.utils_envs import set_up_logger
logger, writer = set_up_logger(opt)

opt.logger = logger
set_up_envs(opt)
opt.cfg.freeze()
# <<<<<<<<<<<<< A bunch of modularised set-ups

semseg_configs = utils_config.load_cfg_from_cfg_file(os.path.join(pwdpath, opt.cfg.MODEL_SEMSEG.config_file))
semseg_configs = utils_config.merge_cfg_from_list(semseg_configs, opt.params)
opt.semseg_configs = semseg_configs

opt.pwdpath = pwdpath

from models_def.model_joint_all import Model_Joint as the_model


# >>>>>>>>>>>>> MODEL AND OPTIMIZER
# build model
# model = MatSeg_BRDF(opt, logger)
model = the_model(opt, logger)
if opt.distributed: # https://github.com/dougsouza/pytorch-sync-batchnorm-example # export NCCL_LL_THRESHOLD=0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(opt.device)
if opt.cfg.MODEL_BRDF.load_pretrained_pth:
    model.load_pretrained_brdf(opt.cfg.MODEL_BRDF.weights)
if opt.cfg.MODEL_SEMSEG.enable and opt.cfg.MODEL_SEMSEG.if_freeze:
    # model.turn_off_names(['UNet'])
    model.turn_off_names(['SEMSEG_Net'])
    model.freeze_bn_semantics()
if opt.cfg.MODEL_MATSEG.enable and opt.cfg.MODEL_MATSEG.if_freeze:
    model.turn_off_names(['MATSEG_Net'])
    model.freeze_bn_matseg()

model.print_net()

# set up optimizers
# optimizer = get_optimizer(model.parameters(), cfg.SOLVER)
optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.lr, betas=(0.5, 0.999) )
if opt.distributed:
    model = DDP(model, device_ids=[opt.rank], output_device=opt.rank, find_unused_parameters=True)

logger.info(red('Optimizer: '+type(optimizer).__name__))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50, cooldown=0, verbose=True, threshold_mode='rel', threshold=0.01)
# <<<<<<<<<<<<< MODEL AND OPTIMIZER

ENABLE_MATSEG = opt.cfg.MODEL_MATSEG.enable
opt.bin_mean_shift_device = opt.device if opt.cfg.MODEL_MATSEG.embed_dims <= 4 else 'cpu'
opt.batch_size_override_vis = -1
if ENABLE_MATSEG:
    if opt.cfg.MODEL_MATSEG.embed_dims > 2:
        opt.batch_size_override_vis = 1      
# opt.batch_size_override_vis = -1 if (opt.bin_mean_shift_device == 'cpu' or not ENABLE_MATSEG) else 1
if opt.cfg.MODEL_MATSEG.embed_dims == 2:
    bin_mean_shift = Bin_Mean_Shift(device=opt.device, invalid_index=opt.invalid_index)
else:
    bin_mean_shift = Bin_Mean_Shift_N(embedding_dims=opt.cfg.MODEL_MATSEG.embed_dims, \
        device=opt.bin_mean_shift_device, invalid_index=opt.invalid_index, if_freeze=opt.cfg.MODEL_MATSEG.if_freeze)
opt.bin_mean_shift = bin_mean_shift

# >>>>>>>>>>>>> DATASET
from utils.utils_semseg import get_transform_semseg, get_transform_matseg, get_transform_resize

transforms_train_semseg = get_transform_semseg('train', opt)
transforms_val_semseg = get_transform_semseg('val', opt)
transforms_train_matseg = get_transform_matseg('train', opt)
transforms_val_matseg = get_transform_matseg('val', opt)
transforms_train_resize = get_transform_resize('train', opt)
transforms_val_resize = get_transform_resize('val', opt)

brdf_dataset_val_vis = openrooms(opt, 
    transforms_fixed = transforms_val_resize, 
    transforms_semseg = transforms_val_semseg, 
    transforms_matseg = transforms_val_matseg,
    transforms_resize = transforms_val_resize, 
    cascadeLevel = opt.cascadeLevel, split = 'val', load_first = opt.cfg.TEST.vis_max_samples, logger=logger)

brdf_loader_val_vis = make_data_loader(
    opt,
    brdf_dataset_val_vis,
    is_train=False,
    start_iter=0,
    logger=logger,
    workers=0,
    batch_size_override=opt.batch_size_override_vis, 
    # pin_memory = False, 
    collate_fn=collate_fn_OR, 
    # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
    if_distributed_override=False
)
# <<<<<<<<<<<<< DATASET

from utils.utils_envs import set_up_checkpointing
checkpointer, tid_start, epoch_start = set_up_checkpointing(opt, model, optimizer, scheduler, logger)

# >>>>>>>>>>>>> TRANING

tid = tid_start
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

ts_iter_end_start_list = []
ts_iter_start_end_list = []
num_mat_masks_MAX = 0

model.train(not opt.cfg.MODEL_SEMSEG.fix_bn)
synchronize()

val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid}
with torch.no_grad():
    vis_val_epoch_joint(brdf_loader_val_vis, model, bin_mean_shift, val_params, batch_size=opt.cfg.TEST.ims_per_batch)