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
os.system('touch %s/models/__init__.py'%pwdpath)
os.system('touch %s/utils/__init__.py'%pwdpath)
print('started.' + pwdpath)

import torchvision.utils as vutils
import utils
from dataset_openrooms import openrooms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T

import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import nvidia_smi

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


# from models.baseline_same import Baseline as UNet
from models.model_mat_seg_brdf import MatSeg_BRDF
from utils.utils_vis import vis_index_map
from utils.config import cfg
from utils.comm import synchronize, get_rank
from utils.misc import AverageMeter, get_optimizer, get_datetime
from utils.bin_mean_shift import Bin_Mean_Shift

from train_funcs import get_input_dict_brdf, train_step
from train_funcs_mat_seg import get_input_dict_mat_seg, forward_mat_seg, val_epoch_mat_seg

from utils.logger import setup_logger, Logger, printer
from utils.global_paths import SUMMARY_PATH, SUMMARY_VIS_PATH, CKPT_PATH
from utils.utils_misc import *
from utils.utils_dataloader import make_data_loader
from utils.utils_training import reduce_loss_dict, check_save, print_gpu_usage, time_meters_to_string
from utils.checkpointer import DetectronCheckpointer
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR


parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--data_root', default=None, help='path to input images')
parser.add_argument('--task_name', type=str, default='tmp', help='task name (e.g. N1: disp ref)')
# The basic training setting
# parser.add_argument('--nepoch0', type=int, default=14, help='the number of epochs for training')
# parser.add_argument('--nepoch1', type=int, default=10, help='the number of epochs for training')

# parser.add_argument('--batchSize0', type=int, default=16, help='input batch size; ALL GPUs')
# parser.add_argument('--batchSize1', type=int, default=16, help='input batch size; ALL GPUs')

# parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to model')
# parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to model')
# parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to model')
# parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to model')

# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1], help='the gpus used for training model')
# Fine tune the model
parser.add_argument('--isFineTune', action='store_true', help='fine-tune the model')
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
parser.add_argument("--master_port", type=str, default='8914')

# DEBUG
parser.add_argument('--debug', action='store_true', help='Debug eval')

parser.add_argument('--ifMatMapInput', action='store_true', help='using mask as additional input')
parser.add_argument('--ifDataloaderOnly', action='store_true', help='benchmark dataloading overhead')
parser.add_argument('--if_cluster', action='store_true', help='if using cluster')
parser.add_argument('--if_hdr', action='store_true', help='if using hdr images')
parser.add_argument('--eval_every_iter', type=int, default=2000, help='')
parser.add_argument('--save_every_iter', type=int, default=2000, help='')
parser.add_argument('--invalid_index', type=int, default = 255, help='index for invalid aread (e.g. windows, lights)')

# Pre-training
parser.add_argument('--resume', type=str, help='resume training; can be full path (e.g. tmp/checkpoint0.pth.tar) or taskname (e.g. tmp); [to continue the current task, use: resume]', default='NoCkpt')
parser.add_argument('--reset_scheduler', action='store_true', help='')
parser.add_argument('--reset_lr', action='store_true', help='')

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
print(opt)
os.environ['MASETER_PORT'] = str(opt.master_port)
cfg.merge_from_file(opt.config_file)
cfg.merge_from_list(opt.params)
cfg.freeze()
opt.cfg = cfg
print(opt.cfg)

# opt.gpuId = opt.deviceIds[0]
# >>>> DISTRIBUTED TRAINING
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

opt.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
opt.distributed = opt.num_gpus > 1
if opt.distributed:
    torch.cuda.set_device(opt.local_rank)
    process_group = torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()
# device = torch.device("cuda" if torch.cuda.is_available() and not opt.cpu else "cpu")
opt.device = 'cuda'
opt.rank = get_rank()
opt.is_master = opt.rank == 0
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(opt.rank)
opt.pwdpath = pwdpath
# <<<< DISTRIBUTED TRAINING

# >>>> SUMMARY WRITERS
if opt.if_cluster:
    opt.home_path = Path('/viscompfs/users/ruizhu/')
    CKPT_PATH = opt.home_path / CKPT_PATH
    SUMMARY_PATH = opt.home_path / SUMMARY_PATH
    SUMMARY_VIS_PATH = opt.home_path / SUMMARY_VIS_PATH
if not opt.if_cluster:
    opt.task_name = get_datetime() + '-' + opt.task_name
opt.summary_path_task = SUMMARY_PATH / opt.task_name
opt.checkpoints_path_task = CKPT_PATH / opt.task_name
opt.summary_vis_path_task = SUMMARY_VIS_PATH / opt.task_name
opt.summary_vis_path_task_py = opt.summary_vis_path_task / 'py_files'

save_folders = [opt.summary_path_task, opt.summary_vis_path_task, opt.summary_vis_path_task_py, opt.checkpoints_path_task, ]
print('====%d/%d', opt.rank, opt.num_gpus, opt.checkpoints_path_task)

if opt.is_master:
    for root_folder in [SUMMARY_PATH, CKPT_PATH, SUMMARY_VIS_PATH]:
        if not root_folder.exists():
            root_folder.mkdir(exist_ok=True)
    if os.path.isdir(opt.summary_path_task):
        if opt.task_name not in ['tmp'] and opt.resume == 'NoCkpt':
            if 'pod' in opt.task_name:            
                if_delete = 'y'
            else:
                if_delete = input(colored('Summary path %s already exists. Delete? [y/n] '%opt.summary_path_task, 'white', 'on_blue'))
                # if_delete = 'y'
            if if_delete == 'y':
                for save_folder in save_folders:
                    os.system('rm -rf %s'%save_folder)
                    print(green('Deleted summary path %s'%save_folder))
    for save_folder in save_folders:
        if not Path(save_folder).is_dir() and opt.rank == 0:
            Path(save_folder).mkdir(exist_ok=True)

synchronize()

# === LOGGING
sys.stdout = Logger(Path(opt.summary_path_task) / 'log.txt')
# sys.stdout = Logger(opt.summary_path_task / 'log.txt')
logger = setup_logger("logger:train", opt.summary_path_task, opt.rank, filename="logger_maskrcn-style.txt")
logger.info(red("==[config]== opt"))
logger.info(opt)
logger.info(red("==[config]== cfg"))
logger.info(cfg)
logger.info(red("==[config]== Loaded configuration file {}".format(opt.config_file)))
with open(opt.config_file, "r") as cf:
    config_str = "\n" + cf.read()
    # logger.info(config_str)
printer = printer(opt.rank, debug=opt.debug)

if opt.is_master and not opt.task_name in ['tmp']:
    exclude_list = ['apex', 'logs_bkg', 'archive', 'train_cifar10_py', 'train_mnist_tf', 'utils_external', 'build/'] + \
        ['Summary', 'Summary_vis', 'Checkpoint', 'logs', '__pycache__', 'snapshots', '.vscode', '.ipynb_checkpoints', 'azureml-setup', 'azureml_compute_logs']
    copy_py_files(opt.pwdpath, opt.summary_vis_path_task_py, exclude_paths=[str(SUMMARY_PATH), str(CKPT_PATH), str(SUMMARY_VIS_PATH)]+exclude_list)
    os.system('cp -r %s %s'%(opt.pwdpath, opt.summary_vis_path_task_py / 'train'))
    logger.info(green('Copied source files %s -> %s'%(opt.pwdpath, opt.summary_vis_path_task_py)))
    # folders = [f for f in Path('./').iterdir() if f.is_dir()]
    # for folder in folders:
    #     folder_dest = opt.summary_vis_path_task_py / folder.name
    #     if not folder_dest.exists() and folder.name not in exclude_list:
    #         os.system('cp -r %s %s'%(folder, folder_dest))

synchronize()

if opt.is_master:
    writer = SummaryWriter(opt.summary_path_task, flush_secs=10)
    print(green('=====>Summary writing to %s'%opt.summary_path_task))
else:
    writer = None
# <<<< SUMMARY WRITERS

# >>>> MODEL AND OPTIMIZER
# build model
model = MatSeg_BRDF(opt)
print('====cfg.MODEL_SEG', cfg.MODEL_SEG)
# if not (opt.resume == 'NoCkpt'):
#     model_dict = torch.load(opt.resume, map_location=lambda storage, loc: storage)
#     model.load_state_dict(model_dict)
if opt.distributed: # https://github.com/dougsouza/pytorch-sync-batchnorm-example # export NCCL_LL_THRESHOLD=0
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = apex.parallel.convert_syncbn_model(model)
model.to(opt.device)

# set up optimizers
optimizer = get_optimizer(model.parameters(), cfg.SOLVER)
# Initialize mixed-precision training
if opt.distributed:
    use_mixed_precision = cfg.MODEL_SEG.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    model = DDP(model)

logger.info(red('Optimizer: '+type(optimizer).__name__))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50, cooldown=0, verbose=True, threshold_mode='rel', threshold=0.01)

# <<<< MODEL AND OPTIMIZER

# >>>> DATASET
transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
brdf_dataset_train = openrooms( opt.data_root, transforms, opt, 
        imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,
        cascadeLevel = opt.cascadeLevel, split = 'train')
# brdf_loader_train = DataLoader(brdf_dataset_train, batch_size = cfg.SOLVER.ims_per_batch,
#         num_workers = 16, shuffle = True, pin_memory=True)
brdf_loader_train = make_data_loader(
    opt,
    brdf_dataset_train,
    is_train=True,
    start_iter=0,
    logger=logger,
    # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
)

if 'mini' in opt.data_root:
    print('=====!!!!===== mini: brdf_dataset_val = brdf_dataset_train')
    brdf_dataset_val = brdf_dataset_train
else:
    brdf_dataset_val = openrooms( opt.data_root, transforms, opt, 
            imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,
            cascadeLevel = opt.cascadeLevel, split = 'val')
# brdfLoaderVal = DataLoader(brdf_dataset_val, batch_size = opt.batchSize,
        # num_workers = 16, shuffle = False, pin_memory=True)
brdf_loader_val = make_data_loader(
    opt,
    brdf_dataset_val,
    is_train=False,
    start_iter=0,
    logger=logger,
    # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
    if_distributed_override=opt.cfg.DATASET.if_val_dist and opt.distributed
)
# <<<< DATASET

# >>>> CHECKPOINTING
save_to_disk = opt.is_master
checkpointer = DetectronCheckpointer(
    opt, model, optimizer, scheduler, CKPT_PATH, opt.checkpoints_path_task, save_to_disk, logger=logger, if_reset_scheduler=opt.reset_scheduler
)
tid_start = 0
epoch_start = 0
if opt.resume != 'NoCkpt':
    if opt.resume == 'resume':
        opt.resume = opt.task_name
    replace_kws = []
    replace_with_kws = []
    # if opt.task_split == 'train':
    #     replace_kws = ['hourglass_model.seq_L2.1', 'hourglass_model.seq_L2.3', 'hourglass_model.disp_res_pred_layer_L2']
    #     replace_with_kws = ['hourglass_model.seq.1', 'hourglass_model.seq.3', 'hourglass_model.disp_res_pred_layer']
    checkpoint_restored, _, _ = checkpointer.load(task_name=opt.resume, replace_kws=replace_kws, replace_with_kws=replace_with_kws)
    if 'iteration' in checkpoint_restored:
        tid_start = checkpoint_restored['iteration']
    if 'epoch' in checkpoint_restored:
        epoch_start = checkpoint_restored['epoch']
    print(checkpoint_restored.keys())
    logger.info(colored('Restoring from epoch %d - iter %d'%(epoch_start, tid_start), 'white', 'on_blue'))
# <<<< CHECKPOINTING

if opt.reset_lr:
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.cfg.SOLVER.lr

# opt.albeW, opt.normW = MODEL_BRDF.albedoWeight, MODEL_BRDF.normalWeight
# opt.rougW = MODEL_BRDF.roughWeight
# opt.deptW = MODEL_BRDF.depthWeight

# if opt.cascadeLevel == 0:
#     opt.nepoch = opt.cfg.SOLVER.max_epoch
#     opt.batchSize = opt.batchSize0
#     opt.imHeight, opt.imWidth = opt.imHeight0, opt.imWidth0
# elif opt.cascadeLevel == 1:
#     opt.nepoch = opt.cfg.SOLVER.max_epoch
#     opt.batchSize = opt.batchSize1
#     opt.imHeight, opt.imWidth = opt.imHeight1, opt.imWidth1

# if opt.experiment is None:
#     opt.experiment = 'check_cascade%d_w%d_h%d' % (opt.cascadeLevel,
#             opt.imWidth, opt.imHeight )

# if opt.if_cluster:
#     opt.experiment = 'logs/' + opt.experiment
# else:
#     opt.experiment = 'logs/' + get_datetime() + opt.experiment
# if opt.if_cluster:
#     opt.experiment = '/viscompfs/users/ruizhu/' + opt.experiment
# os.system('rm -rf {0}'.format(opt.experiment) )
# os.system('mkdir {0}'.format(opt.experiment) )
# os.system('cp -r train %s' % opt.experiment )

# Initial model
# encoder = models_brdf.encoder0(cascadeLevel = opt.cascadeLevel, in_channels = 3 if not opt.ifMatMapInput else 4)
# albedoDecoder = models_brdf.decoder0(mode=0 )
# normalDecoder = models_brdf.decoder0(mode=1 )
# roughDecoder = models_brdf.decoder0(mode=2 )
# depthDecoder = models_brdf.decoder0(mode=4 )
# ####################################################################


# #########################################
# lr_scale = 1
# if opt.isFineTune:
#     print('--- isFineTune=True')
#     encoder.load_state_dict(
#             torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     albedoDecoder.load_state_dict(
#             torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     normalDecoder.load_state_dict(
#             torch.load('{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     roughDecoder.load_state_dict(
#             torch.load('{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     depthDecoder.load_state_dict(
#             torch.load('{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     lr_scale = 1.0 / (2.0 ** (np.floor( ( (opt.epochIdFineTune+1) / 10)  ) ) )
# else:
#     opt.epochIdFineTune = -1
# #########################################
# model = {}
# model['encoder'] = nn.DataParallel(encoder, device_ids = opt.deviceIds )
# model['albedoDecoder'] = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
# model['normalDecoder'] = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
# model['roughDecoder'] = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )
# model['depthDecoder'] = nn.DataParallel(depthDecoder, device_ids = opt.deviceIds )

# ##############  ######################
# # Send things into GPU
# if opt.cuda:
#     model['encoder'] = model['encoder'].cuda(opt.gpuId )
#     model['albedoDecoder'] = model['albedoDecoder'].to(opt.device)
#     model['normalDecoder'] = model['normalDecoder'].to(opt.device)
#     model['roughDecoder'] = model['roughDecoder'].to(opt.device)
#     model['depthDecoder'] = model['depthDecoder'].to(opt.device)
# ####################################


# ####################################
# # Optimizer
# optimizer = {}
# optimizer['opEncoder'] = optim.Adam(model['encoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# optimizer['opAlbedo'] = optim.Adam(model['albedoDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# optimizer['opNormal'] = optim.Adam(model['normalDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# optimizer['opRough'] = optim.Adam(model['roughDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# optimizer['opDepth'] = optim.Adam(model['depthDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# #####################################

# ----------------- Rui from Plane paper 


bin_mean_shift = Bin_Mean_Shift(device=opt.device, invalid_index=opt.invalid_index)

tid = tid_start
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

ts_iter_end_start_list = []
ts_iter_start_end_list = []
num_mat_masks_MAX = 0

model.train(not cfg.MODEL_SEG.fix_bn)
print('=======1', opt.rank)
synchronize()
print('=======2', opt.rank)

# for epoch in list(range(opt.epochIdFineTune+1, opt.cfg.SOLVER.max_epoch)):
# for epoch_0 in list(range(1, 2) ):
for epoch_0 in list(range(opt.cfg.SOLVER.max_epoch)):
    epoch = epoch_0 + epoch_start
    # trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')

    losses = AverageMeter()
    losses_pull = AverageMeter()
    losses_push = AverageMeter()
    losses_binary = AverageMeter()

    time_meters = {}
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    

    epochs_saved = []

    ts_epoch_start = time.time()
    # ts = ts_epoch_start
    # ts_iter_start = ts
    ts_iter_end = ts_epoch_start
    
    print('=======3', opt.rank)
    synchronize()

    for i, data_batch in tqdm(enumerate(brdf_loader_train)):
        reset_tictoc = False
        # Evaluation for an epoch
        if opt.eval_every_iter != -1 and (tid - tid_start) % opt.eval_every_iter == 0:
            val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid}
            val_epoch_mat_seg(brdf_loader_val, model, bin_mean_shift, val_params)
            model.train(not cfg.MODEL_SEG.fix_bn)
            reset_tictoc = True
            
        synchronize()

        # Save checkpoint
        if opt.save_every_iter != -1 and (tid - tid_start) % opt.save_every_iter == 0:
            check_save(opt, tid, tid, epoch, checkpointer, epochs_saved, opt.checkpoints_path_task, logger)
            reset_tictoc = True

        synchronize()

        tid += 1
        if reset_tictoc:
            ts_iter_end = time.time()
        ts_iter_start = time.time()
        if tid > 5:
            ts_iter_end_start_list.append(ts_iter_start - ts_iter_end)

        if opt.ifDataloaderOnly:
            continue

        # ======= Load data from cpu to gpu
        input_dict = get_input_dict_mat_seg(data_batch, opt)
        time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
        time_meters['ts'] = time.time()

        if (tid - tid_start) % 2000 == 0 and opt.is_master:
            for sample_idx in tqdm(range(data_batch['im_not_hdr'].shape[0])):
                # im_single = im_cpu[sample_idx].numpy().squeeze().transpose(1, 2, 0)
                # im_single = im_single**(1.0/2.2)
                im_single = data_batch['im_not_hdr'][sample_idx].numpy().squeeze().transpose(1, 2, 0)

                writer.add_image('TRAIN_im/%d'%sample_idx, im_single, tid, dataformats='HWC')

                mat_aggre_map_single = input_dict['mat_aggre_map_cpu'][sample_idx].numpy().squeeze()
                matAggreMap_single_vis = vis_index_map(mat_aggre_map_single)
                writer.add_image('TRAIN_mat_aggre_map/%d'%sample_idx, matAggreMap_single_vis, tid, dataformats='HWC')

                mat_notlight_mask_single = input_dict['mat_notlight_mask_cpu'][sample_idx].numpy().squeeze()
                writer.add_image('TRAIN_mat_notlight_mask/%d'%sample_idx, mat_notlight_mask_single, tid, dataformats='HW')

                writer.add_text('TRAIN_im_path/%d'%sample_idx, input_dict['im_paths'][sample_idx], tid)
        time_meters['ts'] = time.time()

        # ======= Forward
        output_dict, loss_dict = forward_mat_seg(input_dict, model, opt, time_meters)
        
        # print('=======loss_dict', loss_dict)
        loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
        time_meters['ts'] = time.time()

        # ======= Backward
        optimizer.zero_grad()
        loss = loss_dict['loss_all']
        loss.backward()
        optimizer.step()
        time_meters['backward'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()


        # ======= update loss
        losses.update(loss_dict_reduced['loss_all'].item())
        losses_pull.update(loss_dict_reduced['loss_pull'].item())
        losses_push.update(loss_dict_reduced['loss_push'].item())
        losses_binary.update(loss_dict_reduced['loss_binary'].item())

        if opt.is_master:
            writer.add_scalar('loss_train/loss_all', loss_dict_reduced['loss_all'].item(), tid)
            writer.add_scalar('loss_train/loss_pull', loss_dict_reduced['loss_pull'].item(), tid)
            writer.add_scalar('loss_train/loss_push', loss_dict_reduced['loss_push'].item(), tid)
            writer.add_scalar('loss_train/loss_binary', loss_dict_reduced['loss_binary'].item(), tid)
            writer.add_scalar('training/epoch', epoch, tid)

            logger.info('Epoch %d - Tid %d - loss_all %.3f = loss_pull %.3f + loss_push %.3f + loss_binary %.3f' % \
                (epoch, tid, loss_dict_reduced['loss_all'].item(), loss_dict_reduced['loss_pull'].item(), loss_dict_reduced['loss_push'].item(), loss_dict_reduced['loss_binary'].item()))

        # End of iteration logging
        ts_iter_end = time.time()
        if opt.is_master and (tid - tid_start) > 5:
            ts_iter_start_end_list.append(ts_iter_end - ts_iter_start)
            if (tid - tid_start) % 10 == 0:
                logger.info(green('Rolling end-to-start %.2f, Rolling start-to-end %.2f'%(sum(ts_iter_end_start_list)/len(ts_iter_end_start_list), sum(ts_iter_start_end_list)/len(ts_iter_start_end_list))))
                logger.info(green('Training timings: ' + time_meters_to_string(time_meters)))
            if opt.is_master and tid % 100 == 0:
                print_gpu_usage(handle, logger)

            # print(ts_iter_end_start_list, ts_iter_start_end_list)


        ############# BRDF tmp
        input_batch_brdf, input_dict_brdf, pre_batch_dict_brdf = get_input_dict_brdf(data_batch, opt)
        x1, x2, x3, x4, x5, x6 = model['BRDF_Net']['encoder'](input_batch_brdf)
        albedoPred = 0.5 * (model['BRDF_Net']['albedoDecoder'](input_dict_brdf['imBatch'], x1, x2, x3, x4, x5, x6) + 1)

        # break


    #     albedo_cpu = data_batch['albedo']
    #     albedoBatch = Variable(albedo_cpu ).to(opt.device)

    #     normal_cpu = data_batch['normal']
    #     normalBatch = Variable(normal_cpu ).to(opt.device)

    #     rough_cpu = data_batch['rough']
    #     roughBatch = Variable(rough_cpu ).to(opt.device)

    #     depth_cpu = data_batch['depth']
    #     depthBatch = Variable(depth_cpu ).to(opt.device)


    #     segArea_cpu = data_batch['segArea']
    #     segEnv_cpu = data_batch['segEnv']
    #     segObj_cpu = data_batch['segObj']

    #     seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
    #     segBatch = Variable(seg_cpu ).to(opt.device)

    #     segBRDFBatch = segBatch[:, 2:3, :, :]
    #     segAllBatch = segBatch[:, 0:1, :, :]  + segBatch[:, 2:3, :, :]

    #     # Load the image from cpu to gpu
    #     im_cpu = (data_batch['im'] )
    #     im_batch = Variable(im_cpu ).to(opt.device)


    #     if opt.cascadeLevel > 0:
    #         albedoPre_cpu = data_batch['albedoPre']
    #         albedoPreBatch = Variable(albedoPre_cpu ).to(opt.device)

    #         normalPre_cpu = data_batch['normalPre']
    #         normalPreBatch = Variable(normalPre_cpu ).to(opt.device)

    #         roughPre_cpu = data_batch['roughPre']
    #         roughPreBatch = Variable(roughPre_cpu ).to(opt.device)

    #         depthPre_cpu = data_batch['depthPre']
    #         depthPreBatch = Variable(depthPre_cpu ).to(opt.device)

    #         diffusePre_cpu = data_batch['diffusePre']
    #         diffusePreBatch = Variable(diffusePre_cpu ).to(opt.device)

    #         specularPre_cpu = data_batch['specularPre']
    #         specularPreBatch = Variable(specularPre_cpu ).to(opt.device)

    #         if albedoPreBatch.size(2) < opt.imHeight or albedoPreBatch.size(3) < opt.imWidth:
    #             albedoPreBatch = F.interpolate(albedoPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
    #         if normalPreBatch.size(2) < opt.imHeight or normalPreBatch.size(3) < opt.imWidth:
    #             normalPreBatch = F.interpolate(normalPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
    #         if roughPreBatch.size(2) < opt.imHeight or roughPreBatch.size(3) < opt.imWidth:
    #             roughPreBatch = F.interpolate(roughPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
    #         if depthPreBatch.size(2) < opt.imHeight or depthPreBatch.size(3) < opt.imWidth:
    #             depthPreBatch = F.interpolate(depthPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

    #         # Regress the diffusePred and specular Pred
    #         envRow, envCol = diffusePreBatch.size(2), diffusePreBatch.size(3)
    #         im_batchSmall = F.adaptive_avg_pool2d(im_batch, (envRow, envCol) )
    #         diffusePreBatch, specularPreBatch = models_brdf.LSregressDiffSpec(
    #                 diffusePreBatch, specularPreBatch, im_batchSmall,
    #                 diffusePreBatch, specularPreBatch )

    #         if diffusePreBatch.size(2) < opt.imHeight or diffusePreBatch.size(3) < opt.imWidth:
    #             diffusePreBatch = F.interpolate(diffusePreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
    #         if specularPreBatch.size(2) < opt.imHeight or specularPreBatch.size(3) < opt.imWidth:
    #             specularPreBatch = F.interpolate(specularPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

    #         renderedim_batch = diffusePreBatch + specularPreBatch


    #     # Clear the gradient in optimizer
    #     opEncoder.zero_grad()
    #     opAlbedo.zero_grad()
    #     opNormal.zero_grad()
    #     opRough.zero_grad()
    #     opDepth.zero_grad()

    #     ########################################################
    #     # Build the cascade model architecture #
    #     albedoPreds = []
    #     normalPreds = []
    #     roughPreds = []
    #     depthPreds = []

    #     if opt.cascadeLevel == 0:
    #         if opt.isMatMaskInput:
    #             inputBatch = torch.cat([im_batch, matMaskBatch], dim=1)
    #         else:
    #             inputBatch = im_batch
    #     elif opt.cascadeLevel > 0:
    #         inputBatch = torch.cat([im_batch, albedoPreBatch,
    #             normalPreBatch, roughPreBatch, depthPreBatch,
    #             diffusePreBatch, specularPreBatch], dim=1)

    #     # Initial Prediction
    #     x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    #     albedoPred = 0.5 * (albedoDecoder(im_batch, x1, x2, x3, x4, x5, x6) + 1)
    #     normalPred = normalDecoder(im_batch, x1, x2, x3, x4, x5, x6)
    #     roughPred = roughDecoder(im_batch, x1, x2, x3, x4, x5, x6)
    #     depthPred = 0.5 * (depthDecoder(im_batch, x1, x2, x3, x4, x5, x6 ) + 1)

    #     albedoBatch = segBRDFBatch * albedoBatch
    #     albedoPred = models_brdf.LSregress(albedoPred * segBRDFBatch.expand_as(albedoPred ),
    #             albedoBatch * segBRDFBatch.expand_as(albedoBatch), albedoPred )
    #     albedoPred = torch.clamp(albedoPred, 0, 1)

    #     depthPred = models_brdf.LSregress(depthPred *  segAllBatch.expand_as(depthPred),
    #             depthBatch * segAllBatch.expand_as(depthBatch), depthPred )

    #     albedoPreds.append(albedoPred )
    #     normalPreds.append(normalPred )
    #     roughPreds.append(roughPred )
    #     depthPreds.append(depthPred )

    #     ########################################################

    #     # Compute the error
    #     albedoErrs = []
    #     normalErrs = []
    #     roughErrs = []
    #     depthErrs = []

    #     pixelObjNum = (torch.sum(segBRDFBatch ).cpu().data).item()
    #     pixelAllNum = (torch.sum(segAllBatch ).cpu().data).item()
    #     for n in range(0, len(albedoPreds) ):
    #         albedoErrs.append( torch.sum( (albedoPreds[n] - albedoBatch)
    #             * (albedoPreds[n] - albedoBatch) * segBRDFBatch.expand_as(albedoBatch ) ) / pixelObjNum / 3.0 )
    #     for n in range(0, len(normalPreds) ):
    #         normalErrs.append( torch.sum( (normalPreds[n] - normalBatch)
    #             * (normalPreds[n] - normalBatch) * segAllBatch.expand_as(normalBatch) ) / pixelAllNum / 3.0)
    #     for n in range(0, len(roughPreds) ):
    #         roughErrs.append( torch.sum( (roughPreds[n] - roughBatch)
    #             * (roughPreds[n] - roughBatch) * segBRDFBatch ) / pixelObjNum )
    #     for n in range(0, len(depthPreds ) ):
    #         depthErrs.append( torch.sum( (torch.log(depthPreds[n]+1) - torch.log(depthBatch+1) )
    #             * ( torch.log(depthPreds[n]+1) - torch.log(depthBatch+1) ) * segAllBatch.expand_as(depthBatch ) ) / pixelAllNum )

    #     # Back propagate the gradients
    #     totalErr = 4 * albeW * albedoErrs[-1] + normW * normalErrs[-1] \
    #             + rougW *roughErrs[-1] + deptW * depthErrs[-1]
    #     totalErr.backward()

    #     # Update the model parameter
    #     opEncoder.step()
    #     opAlbedo.step()
    #     opNormal.step()
    #     opRough.step()
    #     opDepth.step()

    #     # Output training error
    #     utils.writeErrToScreen('albedo', albedoErrs, epoch, j )
    #     utils.writeErrToScreen('normal', normalErrs, epoch, j )
    #     utils.writeErrToScreen('rough', roughErrs, epoch, j )
    #     utils.writeErrToScreen('depth', depthErrs, epoch, j )

    #     utils.writeErrToFile('albedo', albedoErrs, trainingLog, epoch, j )
    #     utils.writeErrToFile('normal', normalErrs, trainingLog, epoch, j )
    #     utils.writeErrToFile('rough', roughErrs, trainingLog, epoch, j )
    #     utils.writeErrToFile('depth', depthErrs, trainingLog, epoch, j )

    #     albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0)
    #     normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
    #     roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0)
    #     depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0)

    #     if j < 1000:
    #         utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j)

    #         utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    #     else:
    #         utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), epoch, j)

    #         utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)


    #     if j == 1 or j% 2000 == 0:
    #         # Save the ground truth and the input
    #         vutils.save_image(( (albedoBatch ) ** (1.0/2.2) ).data,
    #                 '{0}/{1}_albedoGt.png'.format(opt.experiment, j) )
    #         vutils.save_image( (0.5*(normalBatch + 1) ).data,
    #                 '{0}/{1}_normalGt.png'.format(opt.experiment, j) )
    #         vutils.save_image( (0.5*(roughBatch + 1) ).data,
    #                 '{0}/{1}_roughGt.png'.format(opt.experiment, j) )
    #         vutils.save_image( ( (im_batch)**(1.0/2.2) ).data,
    #                 '{0}/{1}_im.png'.format(opt.experiment, j) )
    #         depthOut = 1 / torch.clamp(depthBatch + 1, 1e-6, 10) * segAllBatch.expand_as(depthBatch)
    #         vutils.save_image( ( depthOut*segAllBatch.expand_as(depthBatch) ).data,
    #                 '{0}/{1}_depthGt.png'.format(opt.experiment, j) )

    #         if opt.cascadeLevel > 0:
    #             vutils.save_image( ( (diffusePreBatch)**(1.0/2.2) ).data,
    #                     '{0}/{1}_diffusePre.png'.format(opt.experiment, j) )
    #             vutils.save_image( ( (specularPreBatch)**(1.0/2.2) ).data,
    #                     '{0}/{1}_specularPre.png'.format(opt.experiment, j) )
    #             vutils.save_image( ( (renderedim_batch)**(1.0/2.2) ).data,
    #                     '{0}/{1}_renderedImage.png'.format(opt.experiment, j) )

    #         # Save the predicted results
    #         for n in range(0, len(albedoPreds) ):
    #             vutils.save_image( ( (albedoPreds[n] ) ** (1.0/2.2) ).data,
    #                     '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, j, n) )
    #         for n in range(0, len(normalPreds) ):
    #             vutils.save_image( ( 0.5*(normalPreds[n] + 1) ).data,
    #                     '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, j, n) )
    #         for n in range(0, len(roughPreds) ):
    #             vutils.save_image( ( 0.5*(roughPreds[n] + 1) ).data,
    #                     '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, j, n) )
    #         for n in range(0, len(depthPreds) ):
    #             depthOut = 1 / torch.clamp(depthPreds[n] + 1, 1e-6, 10) * segAllBatch.expand_as(depthPreds[n])
    #             vutils.save_image( ( depthOut * segAllBatch.expand_as(depthPreds[n]) ).data,
    #                     '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, n) )

    #     writer.

    # trainingLog.close()

    # # Update the training rate
    # if (epoch + 1) % 10 == 0:
    #     for param_group in opEncoder.param_groups:
    #         param_group['lr'] /= 2
    #     for param_group in opAlbedo.param_groups:
    #         param_group['lr'] /= 2
    #     for param_group in opNormal.param_groups:
    #         param_group['lr'] /= 2
    #     for param_group in opRough.param_groups:
    #         param_group['lr'] /= 2
    #     for param_group in opDepth.param_groups:
    #         param_group['lr'] /= 2
    # # Save the error record
    # np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
    # np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
    # np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
    # np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )

    # # save the models
    # torch.save(encoder.module, '{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    # torch.save(albedoDecoder.module, '{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    # torch.save(normalDecoder.module, '{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    # torch.save(roughDecoder.module, '{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    # torch.save(depthDecoder.module, '{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )

print('num_mat_masks_MAX', num_mat_masks_MAX) 