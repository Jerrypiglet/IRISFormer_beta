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
# from models_def.model_semseg_brdf import SemSeg_BRDF
from models_def.model_joint_all import SemSeg_MatSeg_BRDF as the_model
from train_funcs_joint import get_input_dict_joint, val_epoch_joint, vis_val_epoch_joint, forward_joint, get_time_meters_joint

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

# >>>>>>>>>>>>> A bunch of modularised set-ups
# opt.gpuId = opt.deviceIds[0]
from utils.utils_envs import set_up_dist
handle = set_up_dist(opt)

from utils.utils_envs import set_up_folders
set_up_folders(opt)

from utils.utils_envs import set_up_logger
logger, writer = set_up_logger(opt)
# <<<<<<<<<<<<< A bunch of modularised set-ups

# >>>> MODEL AND OPTIMIZER
# build model
# model = MatSeg_BRDF(opt, logger)
model = the_model(opt, logger)
if opt.distributed: # https://github.com/dougsouza/pytorch-sync-batchnorm-example # export NCCL_LL_THRESHOLD=0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(opt.device)
if opt.cfg.MODEL_BRDF.load_pretrained_pth:
    model.load_pretrained_brdf(opt.cfg.MODEL_BRDF.pretrained_pth_name)
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

# <<<< MODEL AND OPTIMIZER

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

# >>>> DATASET
from utils.utils_semseg import get_transform_semseg, get_transform_matseg
transforms_train_semseg = get_transform_semseg('train', opt)
transforms_val_semseg = get_transform_semseg('val', opt)
transforms_train_matseg = get_transform_matseg('train', opt)
transforms_val_matseg = get_transform_matseg('val', opt)

brdf_dataset_train = openrooms(opt, 
    transforms_fixed = transforms_val_semseg, 
    transforms = transforms_train_semseg, 
    transforms_matseg = transforms_train_matseg,
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
        transforms_fixed = transforms_val_semseg, 
        transforms = transforms_val_semseg, 
        transforms_matseg = transforms_val_matseg,
        # cascadeLevel = opt.cascadeLevel, split = 'val', logger=logger)
        cascadeLevel = opt.cascadeLevel, split = 'val', load_first = 20 if opt.mini_val else -1, logger=logger)
    brdf_dataset_val_vis = openrooms(opt, 
        transforms_fixed = transforms_val_semseg, 
        transforms = transforms_val_semseg, 
        transforms_matseg = transforms_val_matseg,
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
    batch_size_override=opt.batch_size_override_vis, 
    # pin_memory = False, 
    # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
    if_distributed_override=False
)
# <<<< DATASET

from utils.utils_envs import set_up_checkpointing
checkpointer, tid_start, epoch_start = set_up_checkpointing(opt, model, optimizer, scheduler, logger)

# >>>> TRANING

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
    
    print('=======NEW EPOCH', opt.rank, cfg.MODEL_SEMSEG.fix_bn)
    synchronize()

    if tid >= opt.max_iter and opt.max_iter != -1:
        break

    for i, data_batch in enumerate(brdf_loader_train):
        reset_tictoc = False
        # Evaluation for an epoch```
        if opt.eval_every_iter != -1 and (tid - tid_start) % opt.eval_every_iter == 0:
            val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid}
            if opt.if_vis:
                with torch.no_grad():
                    vis_val_epoch_joint(brdf_loader_val_vis, model, bin_mean_shift, val_params)
                synchronize()                
            if opt.if_val:
                with torch.no_grad():
                    val_epoch_joint(brdf_loader_val, model, bin_mean_shift, val_params)
            model.train(not cfg.MODEL_SEMSEG.fix_bn)
            reset_tictoc = True
            
        synchronize()

        # Save checkpoint
        if opt.save_every_iter != -1 and (tid - tid_start) % opt.save_every_iter == 0 and 'tmp' not in opt.task_name:
            check_save(opt, tid, tid, epoch, checkpointer, epochs_saved, opt.checkpoints_path_task, logger)
            reset_tictoc = True

        synchronize()

        torch.cuda.empty_cache()

        if reset_tictoc:
            ts_iter_end = time.time()
        ts_iter_start = time.time()
        if tid > 5:
            ts_iter_end_start_list.append(ts_iter_start - ts_iter_end)

        if opt.ifDataloaderOnly:
            continue
        if tid % opt.debug_every_iter == 0:
            opt.if_vis_debug_pac = True


        # ======= Load data from cpu to gpu
        input_dict = get_input_dict_joint(data_batch, opt)

        time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
        time_meters['ts'] = time.time()

        # ======= Forward
        optimizer.zero_grad()
        output_dict, loss_dict = forward_joint(input_dict, model, opt, time_meters)
        synchronize()
        
        # print('=======loss_dict', loss_dict)
        loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
        time_meters['ts'] = time.time()

        # ======= Backward
        loss = 0.
        loss_keys_backward = []
        loss_keys_print = []
        if opt.cfg.MODEL_MATSEG.enable and (not opt.cfg.MODEL_MATSEG.freeze):
            #  and ((not opt.cfg.MODEL_MATSEG.freeze) or opt.cfg.MODEL_MATSEG.embed_dims <= 4):
            loss_keys_backward.append('loss_matseg-ALL')
            loss_keys_print.append('loss_matseg-ALL')
            loss_keys_print.append('loss_matseg-pull')
            loss_keys_print.append('loss_matseg-push')
            loss_keys_print.append('loss_matseg-binary')

        if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
            loss_keys_backward.append('loss_brdf-ALL')
            loss_keys_print.append('loss_brdf-ALL')
            if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                loss_keys_print.append('loss_brdf-albedo') 
            if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                loss_keys_print.append('loss_brdf-normal') 
            if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                loss_keys_print.append('loss_brdf-rough') 
            if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                loss_keys_print.append('loss_brdf-depth') 

        if (opt.cfg.MODEL_SEMSEG.enable and not opt.cfg.MODEL_SEMSEG.if_freeze) or (opt.cfg.MODEL_BRDF.enable_semseg_decoder):
            loss_keys_backward.append('loss_semseg-ALL')
            loss_keys_print.append('loss_semseg-ALL')
            if opt.cfg.MODEL_SEMSEG.enable:
                loss_keys_print.append('loss_semseg-main') 
                loss_keys_print.append('loss_semseg-aux') 

        loss = sum([loss_dict[loss_key] for loss_key in loss_keys_backward])
        if opt.is_master and tid % 20 == 0:
            print('----loss_dict', loss_dict.keys())
            print('----loss_keys_backward', loss_keys_backward)
        loss.backward()
        optimizer.step()
        time_meters['backward'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()
        synchronize()

        if opt.is_master:
            logger_str = 'Epoch %d - Tid %d -'%(epoch, tid) + ', '.join(['%s %.3f'%(loss_key, loss_dict_reduced[loss_key]) for loss_key in loss_keys_print])
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
        # if opt.is_master:

        if tid % opt.debug_every_iter == 0:       
            if (opt.cfg.MODEL_MATSEG.if_albedo_pooling or opt.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv or opt.cfg.MODEL_MATSEG.if_albedo_pac_pool or opt.cfg.MODEL_MATSEG.if_albedo_safenet) and opt.cfg.MODEL_MATSEG.albedo_pooling_debug:
                if opt.is_master:
                    for sample_idx, im_trainval_RGB_mask_pooled_mean in enumerate(output_dict['im_trainval_RGB_mask_pooled_mean']):
                        im_trainval_RGB_mask_pooled_mean = im_trainval_RGB_mask_pooled_mean.cpu().numpy().squeeze().transpose(1, 2, 0)
                        writer.add_image('TRAIN_im_trainval_RGB_debug/%d'%(sample_idx+(tid*opt.cfg.SOLVER.ims_per_batch)), data_batch['im_trainval_RGB'][sample_idx].numpy().squeeze().transpose(1, 2, 0), tid, dataformats='HWC')
                        writer.add_image('TRAIN_im_trainval_RGB_mask_pooled_mean/%d'%(sample_idx+(tid*opt.cfg.SOLVER.ims_per_batch)), im_trainval_RGB_mask_pooled_mean, tid, dataformats='HWC')
                        logger.info('Added debug pooling sample')
            
        if tid % 2000 == 0:
            for sample_idx, (im_single, im_trainval_RGB, im_path) in enumerate(zip(data_batch['im'], data_batch['im_trainval_RGB'], data_batch['imPath'])):
                im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
                im_trainval_RGB = im_trainval_RGB.numpy().squeeze().transpose(1, 2, 0)
                if opt.is_master:
                    writer.add_image('TRAIN_im/%d'%sample_idx, im_single, tid, dataformats='HWC')
                    writer.add_image('TRAIN_im_trainval_RGB/%d'%sample_idx, im_trainval_RGB, tid, dataformats='HWC')
                    writer.add_text('TRAIN_image_name/%d'%sample_idx, im_path, tid)
            if opt.cfg.DATA.load_matseg_gt:
                for sample_idx, (im_single, mat_aggre_map) in enumerate(zip(data_batch['im_matseg_transformed_trainval'], input_dict['mat_aggre_map_cpu'])):
                    im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
                    mat_aggre_map = mat_aggre_map.numpy().squeeze()
                    if opt.is_master:
                        writer.add_image('TRAIN_matseg_im_trainval/%d'%sample_idx, im_single, tid, dataformats='HWC')
                        writer.add_image('TRAIN_matseg_mat_aggre_map_trainval/%d'%sample_idx, vis_index_map(mat_aggre_map), tid, dataformats='HWC')
                    logger.info('Logged training mat seg')
            if opt.cfg.DATA.load_semseg_gt:
                for sample_idx, (im_single, semseg_label, semseg_pred) in enumerate(zip(data_batch['im_semseg_transformed_trainval'], data_batch['semseg_label'], output_dict['semseg_pred'].detach().cpu())):
                    im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
                    colors = np.loadtxt(os.path.join(opt.pwdpath, opt.cfg.PATH.semseg_colors_path)).astype('uint8')
                    semseg_label = np.uint8(semseg_label.numpy().squeeze())
                    from utils.utils_vis import colorize
                    semseg_label_color = np.array(colorize(semseg_label, colors).convert('RGB'))
                    if opt.is_master:
                        writer.add_image('TRAIN_semseg_im_trainval/%d'%sample_idx, im_single, tid, dataformats='HWC')
                        writer.add_image('TRAIN_semseg_label_trainval/%d'%sample_idx, semseg_label_color, tid, dataformats='HWC')

                    prediction = np.argmax(semseg_pred.numpy().squeeze(), 0)
                    gray_pred = np.uint8(prediction)
                    color_pred = np.array(colorize(gray_pred, colors).convert('RGB'))
                    if opt.is_master:
                        writer.add_image('TRAIN_semseg_PRED/%d'%sample_idx, color_pred, tid, dataformats='HWC')

                    logger.info('Logged training sem seg')



        # if opt.is_master and tid % 100 == 0:
        #     for im_index, (im_single, semseg_color) in enumerate(zip(data_batch['im_not_hdr'], output_dict['semseg_color_list'])):
        #         im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
        #         im_path = os.path.join('./tmp/', 'im_%d-%d_color.png'%(tid, im_index))
        #         color_path = os.path.join('./tmp/', 'im_%d-%d_semseg.png'%(tid, im_index))
        #         cv2.imwrite(im_path, im_single * 255.)
        #         semseg_color.save(color_path)

        # print(input_dict['im_batch_matseg'].shape)
        # print(input_dict['im_batch_matseg'].shape)
        synchronize()
        if tid % opt.debug_every_iter == 0:
            opt.if_vis_debug_pac = False

        tid += 1
        if tid >= opt.max_iter and opt.max_iter != -1:
            break

            # print(ts_iter_end_start_list, ts_iter_start_end_list)