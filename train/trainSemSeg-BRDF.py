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

# from dataset_openrooms import openrooms
from dataset_openroomsV2 import openrooms
from models_def.model_semseg_brdf import SemSeg_BRDF
from train_funcs_joint import get_input_dict_joint, val_epoch_joint, vis_val_epoch_joint, forward_joint, get_time_meters_joint

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.config import cfg
from utils.bin_mean_shift import Bin_Mean_Shift
from utils.comm import synchronize
from utils.utils_misc import *
from utils.utils_dataloader import make_data_loader
from utils.utils_training import reduce_loss_dict, check_save, print_gpu_usage, time_meters_to_string, find_free_port
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR

import utils.utils_config as utils_config
from utils.utils_envs import set_up_envs

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
parser.add_argument('--if_hdr_input_mat_seg', action='store_true', help='if using hdr images')
parser.add_argument('--eval_every_iter', type=int, default=2000, help='')
parser.add_argument('--save_every_iter', type=int, default=2000, help='')
parser.add_argument('--invalid_index', type=int, default = 255, help='index for invalid aread (e.g. windows, lights)')

# Pre-training
parser.add_argument('--resume', type=str, help='resume training; can be full path (e.g. tmp/checkpoint0.pth.tar) or taskname (e.g. tmp); [to continue the current task, use: resume]', default='NoCkpt')
parser.add_argument('--reset_scheduler', action='store_true', help='')
parser.add_argument('--reset_lr', action='store_true', help='')

# to get rid of
parser.add_argument('--test_real', action='store_true', help='')

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
set_up_envs(opt)
opt.cfg.freeze()

semseg_configs = utils_config.load_cfg_from_cfg_file(os.path.join(pwdpath, opt.cfg.MODEL_SEMSEG.config_file))
semseg_configs = utils_config.merge_cfg_from_list(semseg_configs, opt.params)
opt.semseg_configs = semseg_configs

opt.pwdpath = pwdpath

# opt.gpuId = opt.deviceIds[0]
from utils.utils_envs import set_up_dist
handle = set_up_dist(opt)

from utils.utils_envs import set_up_folders
set_up_folders(opt)

from utils.utils_envs import set_up_logger
logger, writer = set_up_logger(opt)

# >>>> MODEL AND OPTIMIZER
# build model
# model = MatSeg_BRDF(opt, logger)
model = SemSeg_BRDF(opt, logger)
# if not (opt.resume == 'NoCkpt'):
#     model_dict = torch.load(opt.resume, map_location=lambda storage, loc: storage)
#     model.load_state_dict(model_dict)
if opt.distributed: # https://github.com/dougsouza/pytorch-sync-batchnorm-example # export NCCL_LL_THRESHOLD=0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(opt.device)
model.print_net()
if opt.cfg.MODEL_BRDF.load_pretrained_pth:
    model.load_pretrained_brdf(opt.cfg.MODEL_BRDF.pretrained_pth_name)
if opt.cfg.MODEL_SEMSEG.enable and opt.cfg.MODEL_SEMSEG.if_freeze:
    model.turn_off_names(['UNet'])
    model.freeze_bn_semantics()

# set up optimizers
# optimizer = get_optimizer(model.parameters(), cfg.SOLVER)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999) )
if opt.distributed:
    model = DDP(model, device_ids=[opt.rank], output_device=opt.rank)

logger.info(red('Optimizer: '+type(optimizer).__name__))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50, cooldown=0, verbose=True, threshold_mode='rel', threshold=0.01)

# <<<< MODEL AND OPTIMIZER

# >>>> DATASET
from utils.utils_semseg import get_transform_semseg
transforms_semseg_train = get_transform_semseg('train', opt)
transforms_semseg_val = get_transform_semseg('val', opt)

brdf_dataset_train = openrooms(opt, 
    transforms_fixed = transforms_semseg_val, 
    transforms_semseg = transforms_semseg_train, 
    cascadeLevel = opt.cascadeLevel, split = 'train', logger=logger)
brdf_loader_train = make_data_loader(
    opt,
    brdf_dataset_train,
    is_train=True,
    start_iter=0,
    logger=logger,
    # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
)
if opt.cfg.MODEL_SEMSEG.enable:
    opt.semseg_colors = brdf_dataset_train.semseg_colors

if 'mini' in opt.cfg.DATASET.dataset_path:
    print('=====!!!!===== mini: brdf_dataset_val = brdf_dataset_train')
    brdf_dataset_val = brdf_dataset_train
    brdf_dataset_val_vis = brdf_dataset_train
else:
    brdf_dataset_val = openrooms(opt, 
        transforms_fixed = transforms_semseg_val, 
        transforms_semseg = transforms_semseg_val, 
        cascadeLevel = opt.cascadeLevel, split = 'val', logger=logger)
    brdf_dataset_val_vis = openrooms(opt, 
        transforms_fixed = transforms_semseg_val, 
        transforms_semseg = transforms_semseg_val, 
        cascadeLevel = opt.cascadeLevel, split = 'val', load_first = 20, logger=logger)
brdf_loader_val = make_data_loader(
    opt,
    brdf_dataset_val,
    is_train=False,
    start_iter=0,
    logger=logger,
    # pin_memory = False, 
    # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
    if_distributed_override=opt.cfg.DATASET.if_val_dist and opt.distributed
)
brdf_loader_val_vis = make_data_loader(
    opt,
    brdf_dataset_val_vis,
    is_train=False,
    start_iter=0,
    logger=logger,
    workers=0,
    # pin_memory = False, 
    # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
    if_distributed_override=False
)
# <<<< DATASET

from utils.utils_envs import set_up_checkpointing
checkpointer, tid_start, epoch_start = set_up_checkpointing(opt, model, optimizer, scheduler, logger)

# >>>> TRANING
bin_mean_shift = Bin_Mean_Shift(device=opt.device, invalid_index=opt.invalid_index)

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

# if task_split

# for epoch in list(range(opt.epochIdFineTune+1, opt.cfg.SOLVER.max_epoch)):
# for epoch_0 in list(range(1, 2) ):
for epoch_0 in list(range(opt.cfg.SOLVER.max_epoch)):
    epoch = epoch_0 + epoch_start
    # trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')

    # losses = AverageMeter()
    # losses_pull = AverageMeter()
    # losses_push = AverageMeter()
    # losses_binary = AverageMeter()

    time_meters = get_time_meters_joint()


    epochs_saved = []

    ts_epoch_start = time.time()
    # ts = ts_epoch_start
    # ts_iter_start = ts
    ts_iter_end = ts_epoch_start
    
    print('=======3', opt.rank, cfg.MODEL_SEMSEG.fix_bn)
    synchronize()

    for i, data_batch in enumerate(brdf_loader_train):
        reset_tictoc = False
        # Evaluation for an epoch```
        if opt.eval_every_iter != -1 and (tid - tid_start) % opt.eval_every_iter == 0:
            val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid}
            if opt.if_vis:
                vis_val_epoch_joint(brdf_loader_val_vis, model, bin_mean_shift, val_params)
                synchronize()                
            if opt.if_val:
                val_epoch_joint(brdf_loader_val, model, bin_mean_shift, val_params)
            model.train(not cfg.MODEL_SEMSEG.fix_bn)
            reset_tictoc = True
            
        synchronize()

        # Save checkpoint
        if opt.save_every_iter != -1 and (tid - tid_start) % opt.save_every_iter == 0:
            check_save(opt, tid, tid, epoch, checkpointer, epochs_saved, opt.checkpoints_path_task, logger)
            reset_tictoc = True

        synchronize()

        if reset_tictoc:
            ts_iter_end = time.time()
        ts_iter_start = time.time()
        if tid > 5:
            ts_iter_end_start_list.append(ts_iter_start - ts_iter_end)

        if opt.ifDataloaderOnly:
            continue

        # ======= Load data from cpu to gpu
        input_dict = get_input_dict_joint(data_batch, opt)

        time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
        time_meters['ts'] = time.time()

        # ======= Forward
        optimizer.zero_grad()
        output_dict, loss_dict = forward_joint(input_dict, model, opt, time_meters)
        
        # print('=======loss_dict', loss_dict)
        loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
        time_meters['ts'] = time.time()

        # ======= Backward
        loss = 0.
        loss_keys_backward = []
        if opt.cfg.MODEL_SEG.enable:
            loss_keys_backward.append('loss_mat_seg-ALL')
        if opt.cfg.MODEL_BRDF.enable:
            loss_keys_backward.append('loss_brdf-ALL')
        if opt.cfg.MODEL_SEMSEG.enable and not opt.cfg.MODEL_SEMSEG.if_freeze:
            loss_keys_backward.append('loss_semseg-ALL')
        loss = sum([loss_dict[loss_key] for loss_key in loss_keys_backward])
        loss.backward()
        optimizer.step()
        time_meters['backward'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

        if opt.is_master:
            logger_str = 'Epoch %d - Tid %d -'%(epoch, tid) + ', '.join(['%s %.3f'%(loss_key, loss_dict_reduced[loss_key]) for loss_key in loss_keys_backward])
            logger.info(logger_str)

            for loss_key in loss_dict_reduced:
                writer.add_scalar('loss_train/%s'%loss_key, loss_dict_reduced[loss_key].item(), tid)
            writer.add_scalar('training/epoch', epoch, tid)

        # End of iteration logging
        ts_iter_end = time.time()
        if opt.is_master and (tid - tid_start) > 5:
            ts_iter_start_end_list.append(ts_iter_end - ts_iter_start)
            if (tid - tid_start) % 10 == 0:
                logger.info(green('Rolling end-to-start %.2f, Rolling start-to-end %.2f'%(sum(ts_iter_end_start_list)/len(ts_iter_end_start_list), sum(ts_iter_start_end_list)/len(ts_iter_start_end_list))))
                logger.info(green('Training timings: ' + time_meters_to_string(time_meters)))
            if opt.is_master and tid % 100 == 0:
                usage_ratio = print_gpu_usage(handle, logger)
                writer.add_scalar('training/GPU_usage_ratio', usage_ratio, tid)
                writer.add_scalar('training/batch_size_per_gpu', len(data_batch['imPath']), tid)
                writer.add_scalar('training/gpus', opt.num_gpus, tid)
            if opt.is_master and tid % 1000 == 0:
                for sample_idx, (im_single, im_path) in enumerate(zip(data_batch['im'], data_batch['im_trainval_RGB'], data_batch['imPath'])):
                    im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
                    writer.add_image('TRAIN_im/%d'%sample_idx, im_single, tid, dataformats='HWC')
                    writer.add_text('TRAIN_image_name/%d'%sample_idx, im_path, tid)


        # if opt.is_master and tid % 100 == 0:
        #     for im_index, (im_single, semseg_color) in enumerate(zip(data_batch['im_not_hdr'], output_dict['semseg_color_list'])):
        #         im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
        #         im_path = os.path.join('./tmp/', 'im_%d-%d_color.png'%(tid, im_index))
        #         color_path = os.path.join('./tmp/', 'im_%d-%d_semseg.png'%(tid, im_index))
        #         cv2.imwrite(im_path, im_single * 255.)
        #         semseg_color.save(color_path)

        tid += 1

            # print(ts_iter_end_start_list, ts_iter_start_end_list)