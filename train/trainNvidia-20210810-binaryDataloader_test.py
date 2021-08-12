import torch
# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.benchmark = True
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
# import random
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

# from dataset_openroomsV4_total3d_matcls_ import openrooms, collate_fn_OR
from dataset_openrooms_OR_scanNetPose import openrooms, collate_fn_OR
from dataset_openrooms_OR_scanNetPose_binary_test import openrooms_binary
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.config import cfg
from utils.bin_mean_shift import Bin_Mean_Shift
from utils.bin_mean_shift_3 import Bin_Mean_Shift_3
from utils.bin_mean_shift_N import Bin_Mean_Shift_N
from utils.comm import synchronize
from utils.utils_misc import *
# from utils.utils_dataloader import make_data_loader
from utils.utils_dataloader_binary import make_data_loader_binary
from utils.utils_training import reduce_loss_dict, check_save, print_gpu_usage, time_meters_to_string, find_free_port
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR

import utils.utils_config as utils_config
from utils.utils_envs import set_up_envs
from utils.utils_total3D.utils_others import OR4XCLASSES_dict
from utils.utils_vis import vis_index_map
from icecream import ic

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--data_root', default=None, help='path to input images')
parser.add_argument('--task_name', type=str, default='tmp', help='task name (e.g. N1: disp ref)')
parser.add_argument('--task_split', type=str, default='train', help='train, val, test', choices={"train", "val", "test"})
# Fine tune the model
parser.add_argument('--isFineTune', action='store_true', help='fine-tune the model')
parser.add_argument("--if_train", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_val", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_vis", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--if_overfit_val", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--if_overfit_train", type=str2bool, nargs='?', const=True, default=False)
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
# parser.add_argument('--ifDataloaderOnly', action='store_true', help='benchmark dataloading overhead')
parser.add_argument('--if_cluster', action='store_true', help='if using cluster')
parser.add_argument('--cluster', type=str, default='kubectl', help='cluster name if if_cluster is True', choices={"kubectl", "nvidia", "ngc"})
parser.add_argument('--if_hdr_input_matseg', action='store_true', help='if using hdr images')
parser.add_argument('--eval_every_iter', type=int, default=2000, help='')
parser.add_argument('--save_every_iter', type=int, default=5000, help='')
parser.add_argument('--debug_every_iter', type=int, default=2000, help='')
parser.add_argument('--max_iter', type=int, default=-1, help='')
parser.add_argument('--invalid_index', type=int, default = 0, help='index for invalid aread (e.g. windows, lights)')

# Pre-training
parser.add_argument('--resume', type=str, help='resume training; can be full path (e.g. tmp/checkpoint0.pth.tar) or taskname (e.g. tmp); [to continue the current task, use: resume]', default='NoCkpt')
parser.add_argument('--resumes_extra', type=str, help='list of extra resumed checkpoints; strings concat by #', default='NoCkpt')
parser.add_argument('--reset_latest_ckpt', action='store_true', help='remove latest_checkpoint file')
parser.add_argument('--reset_scheduler', action='store_true', help='')
parser.add_argument('--reset_lr', action='store_true', help='')
parser.add_argument('--reset_tid', action='store_true', help='')
# debug
# parser.add_argument("--mini_val", type=str2bool, nargs='?', const=True, default=False)
# to get rid of
parser.add_argument('--test_real', action='store_true', help='')

parser.add_argument('--skip_keys', nargs='+', help='Skip those keys in the model', required=False)
parser.add_argument('--replaced_keys', nargs='+', help='Replace those keys in the model', required=False)
parser.add_argument('--replacedby', nargs='+', help='... with those keys from ckpt. Must be in the same length as ``replace_leys``', required=False)
parser.add_argument("--if_save_pickles", type=str2bool, nargs='?', const=True, default=False)

parser.add_argument('--meta_splits_skip', nargs='+', help='Skip those keys in the model', required=False)

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
opt.pwdpath = pwdpath

# >>>>>>>>>>>>> A bunch of modularised set-ups
# opt.gpuId = opt.deviceIds[0]
semseg_configs = utils_config.load_cfg_from_cfg_file(os.path.join(pwdpath, opt.cfg.MODEL_SEMSEG.config_file))
semseg_configs = utils_config.merge_cfg_from_list(semseg_configs, opt.params)
opt.semseg_configs = semseg_configs

from utils.utils_envs import set_up_dist
handle = set_up_dist(opt)

from utils.utils_envs import set_up_folders
set_up_folders(opt)

from utils.utils_envs import set_up_logger
logger, writer = set_up_logger(opt)

opt.logger = logger
set_up_envs(opt)
opt.cfg.freeze()

if opt.is_master:
    ic(opt.cfg)
# <<<<<<<<<<<<< A bunch of modularised set-ups

# >>>>>>>>>>>>> DATASET
from utils.utils_semseg import get_transform_semseg, get_transform_matseg, get_transform_resize

transforms_train_semseg = get_transform_semseg('train', opt)
transforms_val_semseg = get_transform_semseg('val', opt)
transforms_train_matseg = get_transform_matseg('train', opt)
transforms_val_matseg = get_transform_matseg('val', opt)
transforms_train_resize = get_transform_resize('train', opt)
transforms_val_resize = get_transform_resize('val', opt)

if opt.if_train:
    brdf_dataset_train = openrooms_binary(opt, 
        transforms_fixed = transforms_val_resize, 
        transforms_semseg = transforms_train_semseg, 
        transforms_matseg = transforms_train_matseg,
        transforms_resize = transforms_train_resize, 
        cascadeLevel = opt.cascadeLevel, split = 'train', if_for_training=True, logger=logger)
    brdf_loader_train, _ = make_data_loader_binary(
        opt,
        brdf_dataset_train,
        is_train=True,
        start_iter=0,
        logger=logger,
        collate_fn=collate_fn_OR, 
        # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
)

# for epoch_0 in range(2):
#     print('=======NEW EPOCH', opt.rank, cfg.MODEL_SEMSEG.fix_bn)
synchronize()

frame_list = []
for i, data_batch in tqdm(enumerate(brdf_loader_train)):
    if opt.rank==0:
        print('--', i, opt.rank, len(data_batch['frame_info']), len(brdf_dataset_train.scene_key_frame_id_list_this_rank))
    # print(data_batch)
    scene_key_list, frame_id_list = [_['scene_key'] for _ in data_batch['frame_info']], [_['frame_id'] for _ in data_batch['frame_info']]

    # frame_list += [' '.join([x, str(y)]) for x, y in zip(scene_key_list, frame_id_list)]
    frame_list += [[x, y] for x, y in zip(scene_key_list, frame_id_list)]
    synchronize()

print('---->', opt.rank, frame_list[:10], len(frame_list), len(brdf_dataset_train.scene_key_frame_id_list_this_rank), len(brdf_dataset_train.scene_key_frame_id_list))
print('---->', opt.rank, [x for x in frame_list if x not in brdf_dataset_train.scene_key_frame_id_list_this_rank])
# synchronize()

# if opt.distributed:
#     from train_funcs_detectron import gather_lists
#     frame_list_gathered = gather_lists(frame_list, opt.num_gpus)
#     print(len(frame_list_gathered))

#     print(len(brdf_dataset_train.scene_key_frame_id_list))
