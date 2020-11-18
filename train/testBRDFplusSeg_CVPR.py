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

import torchvision.utils as vutils
import utils
from dataset_openrooms import openrooms
from dataset_openrooms_real import openrooms_real
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
import cv2


# from models_def.baseline_same import Baseline as UNet
from models_def.model_brdf_plus_semseg_cvpr import BRDFplusSemSeg
from utils.utils_vis import vis_index_map
from utils.config import cfg
from utils.comm import synchronize, get_rank
from utils.utils_training import get_optimizer, freeze_bn_in_module
from utils.bin_mean_shift import Bin_Mean_Shift

from train_funcs_joint import get_input_dict_joint, val_epoch_joint, vis_val_epoch_joint, forward_joint, get_time_meters_joint

from utils.logger import setup_logger, Logger, printer
from utils.global_paths import SUMMARY_PATH, SUMMARY_VIS_PATH, CKPT_PATH
from utils.utils_misc import *
from utils.utils_dataloader import make_data_loader
from utils.utils_training import reduce_loss_dict, check_save, print_gpu_usage, time_meters_to_string
from utils.checkpointer import DetectronCheckpointer
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR

import utils.utils_config as utils_config


parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--data_root', default=None, help='path to input images')
parser.add_argument('--task_name', type=str, default='tmp', help='task name (e.g. N1: disp ref)')
parser.add_argument('--task_split', type=str, default='train', help='train, val, test', choices={"train", "val", "test"})
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
parser.add_argument("--master_port", type=str, default='8914')

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

parser.add_argument('--test_real', action='store_true', help='')
parser.add_argument("--real_list", type=str, default='')


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
os.environ['MASETER_PORT'] = str(nextPort(int(opt.master_port)))
cfg.merge_from_file(opt.config_file)
cfg.merge_from_list(opt.params)


cfg.freeze()
opt.cfg = cfg
print(opt.cfg)

# semseg_configs = utils_config.load_cfg_from_cfg_file(os.path.join(pwdpath, cfg.MODEL_SEMSEG.config_file))
# semseg_configs = utils_config.merge_cfg_from_list(semseg_configs, opt.params)
# opt.semseg_configs = semseg_configs

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
    # synchronize()
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
            if 'POD' in opt.task_name:            
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
# model = MatSeg_BRDF(opt, logger)
model = BRDFplusSemSeg(opt, logger)
print('====cfg.MODEL_SEMSEG', cfg.MODEL_SEMSEG)
# if not (opt.resume == 'NoCkpt'):
#     model_dict = torch.load(opt.resume, map_location=lambda storage, loc: storage)
#     model.load_state_dict(model_dict)
if opt.distributed: # https://github.com/dougsouza/pytorch-sync-batchnorm-example # export NCCL_LL_THRESHOLD=0
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = apex.parallel.convert_syncbn_model(model)
model.to(opt.device)
model.print_net()
if opt.cfg.MODEL_BRDF.load_pretrained_pth:
    # print(opt.cfg.MODEL_BRDF.pre_trained_Zhengqin, '================----')
    model.load_pretrained_brdf(opt.cfg.MODEL_BRDF.pretrained_pth_name)
if opt.cfg.MODEL_SEMSEG.enable and opt.cfg.MODEL_SEMSEG.if_freeze:
    # print(opt.cfg.MODEL_SEMSEG.if_freeze, dtype(opt.cfg.MODEL_SEMSEG.if_freeze), '================')
    model.turn_off_names(['UNet'])
    model.freeze_bn_semantics()


# set up optimizers
# optimizer = get_optimizer(model.parameters(), cfg.SOLVER)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999) )
# Initialize mixed-precision training
if opt.distributed:
    use_mixed_precision = cfg.DTYPE == "float16"
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

# brdf_dataset_train = openrooms( opt.data_root, transforms, opt, 
#         imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,
#         cascadeLevel = opt.cascadeLevel, split = 'train')
# # brdf_loader_train = DataLoader(brdf_dataset_train, batch_size = cfg.SOLVER.ims_per_batch,
# #         num_workers = 16, shuffle = True, pin_memory=True)
# brdf_loader_train = make_data_loader(
#     opt,
#     brdf_dataset_train,
#     is_train=True,
#     start_iter=0,
#     logger=logger,
#     # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
# )
if opt.cfg.MODEL_SEMSEG.enable:
    opt.semseg_colors = brdf_dataset_train.semseg_colors

# if 'mini' in opt.data_root:
#     print('=====!!!!===== mini: brdf_dataset_val = brdf_dataset_train')
#     brdf_dataset_val = brdf_dataset_train
#     brdf_dataset_val_vis = brdf_dataset_train
# else:
    # brdf_dataset_val = openrooms( opt.data_root, transforms, opt, 
    #         imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,
    #         cascadeLevel = opt.cascadeLevel, split = 'test', phase = 'TEST')
    # brdf_dataset_val_vis = openrooms( opt.data_root, transforms, opt, 
    #         imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,
    #         cascadeLevel = opt.cascadeLevel, split = 'test', phase = 'TEST')

if opt.test_real:
    brdf_dataset_val = openrooms_real( opt.data_root, transforms, opt, opt.real_list, 
            imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,)
    brdf_dataset_val_vis = openrooms_real( opt.data_root, transforms, opt, opt.real_list, 
            imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,)
else:    
    # brdf_dataset_val = openrooms( opt.data_root, transforms, opt, 
    #         imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,
    #         cascadeLevel = opt.cascadeLevel, split = 'val')
    # brdf_dataset_val_vis = openrooms( opt.data_root, transforms, opt, 
    #         imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,
    #         cascadeLevel = opt.cascadeLevel, split = 'val')

    brdf_dataset_val = openrooms( opt.data_root, transforms, opt, 
            imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,
            cascadeLevel = opt.cascadeLevel, split = 'test', phase = 'TEST')
    brdf_dataset_val_vis = openrooms( opt.data_root, transforms, opt, 
            imWidth = opt.cfg.DATA.im_width, imHeight = opt.cfg.DATA.im_height,
            cascadeLevel = opt.cascadeLevel, split = 'test', phase = 'TEST')




# brdfLoaderVal = DataLoader(brdf_dataset_val, batch_size = opt.batchSize,
        # num_workers = 16, shuffle = False, pin_memory=True)
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
    brdf_dataset_val,
    is_train=False,
    start_iter=0,
    logger=logger,
    workers=0,
    # pin_memory = False, 
    # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
    if_distributed_override=False
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
    # if 'train_POD_matseg_DDP' in opt.resume:
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

model.train(not cfg.MODEL_SEMSEG.fix_bn)
print('=======1', opt.rank)
synchronize()
print('=======2', opt.rank)

if cfg.MODEL_BRDF.enable_semseg_decoder:
    opt.semseg_criterion = nn.CrossEntropyLoss(ignore_index=opt.cfg.MODEL_BRDF.semseg_ignore_label)


val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid}
if opt.if_vis:
    vis_val_epoch_joint(brdf_loader_val_vis, model, bin_mean_shift, val_params)
    synchronize()                
if opt.if_val and not opt.test_real:
    val_epoch_joint(brdf_loader_val, model, bin_mean_shift, val_params)



# # if task_split

# # for epoch in list(range(opt.epochIdFineTune+1, opt.cfg.SOLVER.max_epoch)):
# # for epoch_0 in list(range(1, 2) ):
# for epoch_0 in list(range(opt.cfg.SOLVER.max_epoch)):
#     epoch = epoch_0 + epoch_start
#     # trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')

#     # losses = AverageMeter()
#     # losses_pull = AverageMeter()
#     # losses_push = AverageMeter()
#     # losses_binary = AverageMeter()

#     time_meters = get_time_meters_joint()


#     epochs_saved = []

#     ts_epoch_start = time.time()
#     # ts = ts_epoch_start
#     # ts_iter_start = ts
#     ts_iter_end = ts_epoch_start
    
#     print('=======3', opt.rank, not cfg.MODEL_SEMSEG.fix_bn)
#     synchronize()

#     for i, data_batch in enumerate(brdf_loader_train):
#         reset_tictoc = False
#         # Evaluation for an epoch```
#         if opt.eval_every_iter != -1 and (tid - tid_start) % opt.eval_every_iter == 0:
#             val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid}
#             if opt.if_vis:
#                 vis_val_epoch_joint(brdf_loader_val_vis, model, bin_mean_shift, val_params)
#                 synchronize()                
#             if opt.if_val:
#                 val_epoch_joint(brdf_loader_val, model, bin_mean_shift, val_params)
#             model.train(not cfg.MODEL_SEMSEG.fix_bn)
#             reset_tictoc = True
            
#         synchronize()

#         # Save checkpoint
#         if opt.save_every_iter != -1 and (tid - tid_start) % opt.save_every_iter == 0:
#             check_save(opt, tid, tid, epoch, checkpointer, epochs_saved, opt.checkpoints_path_task, logger)
#             reset_tictoc = True

#         synchronize()

#         if reset_tictoc:
#             ts_iter_end = time.time()
#         ts_iter_start = time.time()
#         if tid > 5:
#             ts_iter_end_start_list.append(ts_iter_start - ts_iter_end)

#         if opt.ifDataloaderOnly:
#             continue

#         # ======= Load data from cpu to gpu
#         input_dict = get_input_dict_joint(data_batch, opt)

#         time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
#         time_meters['ts'] = time.time()

#         # if (tid - tid_start) % 2000 == 0 and opt.is_master:
#         #     for sample_idx in tqdm(range(data_batch['im_not_hdr'].shape[0])):
#         #         # im_single = im_cpu[sample_idx].numpy().squeeze().transpose(1, 2, 0)
#         #         # im_single = im_single**(1.0/2.2)
#         #         im_single = data_batch['im_not_hdr'][sample_idx].numpy().squeeze().transpose(1, 2, 0)

#         #         writer.add_image('TRAIN_im/%d'%sample_idx, im_single, tid, dataformats='HWC')

#         #         mat_aggre_map_single = input_dict['mat_aggre_map_cpu'][sample_idx].numpy().squeeze()
#         #         matAggreMap_single_vis = vis_index_map(mat_aggre_map_single)
#         #         writer.add_image('TRAIN_mat_aggre_map/%d'%sample_idx, matAggreMap_single_vis, tid, dataformats='HWC')

#         #         mat_notlight_mask_single = input_dict['mat_notlight_mask_cpu'][sample_idx].numpy().squeeze()
#         #         writer.add_image('TRAIN_mat_notlight_mask/%d'%sample_idx, mat_notlight_mask_single, tid, dataformats='HW')

#         #         writer.add_text('TRAIN_im_path/%d'%sample_idx, input_dict['im_paths'][sample_idx], tid)
#         # time_meters['ts'] = time.time()

#         # ======= Forward
#         optimizer.zero_grad()
#         output_dict, loss_dict = forward_joint(input_dict, model, opt, time_meters)
        
#         # print('=======loss_dict', loss_dict)
#         loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
#         time_meters['ts'] = time.time()

#         # ======= Backward
#         loss = 0.
#         if opt.cfg.MODEL_SEG.enable:
#             loss += loss_dict['loss_mat_seg-ALL']
#         if opt.cfg.MODEL_BRDF.enable:
#             loss += loss_dict['loss_brdf-ALL']
#             if opt.cfg.MODEL_BRDF.enable_semseg_decoder:
#                 loss += loss_dict['loss_brdf-semseg']

#         loss.backward()
#         optimizer.step()
#         time_meters['backward'].update(time.time() - time_meters['ts'])
#         time_meters['ts'] = time.time()


#         # # ======= update loss
#         # losses.update(loss_dict_reduced['loss_all'].item())
#         # losses_pull.update(loss_dict_reduced['loss_pull'].item())
#         # losses_push.update(loss_dict_reduced['loss_push'].item())
#         # losses_binary.update(loss_dict_reduced['loss_binary'].item())

#         if opt.is_master:
#             for loss_key in loss_dict_reduced:
#                 # print(loss_key, loss_dict_reduced[loss_key].item())
#                 writer.add_scalar('loss_train/%s'%loss_key, loss_dict_reduced[loss_key].item(), tid)
#             writer.add_scalar('training/epoch', epoch, tid)

#             # if opt.cfg.MODEL_SEMSEG.enable:
#             #     logger.info('Epoch %d - Tid %d - loss_mat_seg-ALL %.3f = loss_pull %.3f + loss_push %.3f + loss_binary %.3f' % \
#             #         (epoch, tid, loss_dict_reduced['loss_mat_seg-ALL'].item(), loss_dict_reduced['loss_mat_seg-pull'].item(), loss_dict_reduced['loss_mat_seg-push'].item(), loss_dict_reduced['loss_mat_seg-binary'].item()))
                    
#             if opt.cfg.MODEL_BRDF.enable:
#                 logger.info('Epoch %d - Tid %d - loss_brdf-ALL %.3f' % \
#                     (epoch, tid, loss_dict_reduced['loss_brdf-ALL'].item()))

#         # End of iteration logging
#         ts_iter_end = time.time()
#         if opt.is_master and (tid - tid_start) > 5:
#             ts_iter_start_end_list.append(ts_iter_end - ts_iter_start)
#             if (tid - tid_start) % 10 == 0:
#                 logger.info(green('Rolling end-to-start %.2f, Rolling start-to-end %.2f'%(sum(ts_iter_end_start_list)/len(ts_iter_end_start_list), sum(ts_iter_start_end_list)/len(ts_iter_start_end_list))))
#                 logger.info(green('Training timings: ' + time_meters_to_string(time_meters)))
#             if opt.is_master and tid % 100 == 0:
#                 usage_ratio = print_gpu_usage(handle, logger)
#                 writer.add_scalar('training/GPU_usage_ratio', usage_ratio, tid)
#                 writer.add_scalar('training/batch_size_per_gpu', len(data_batch['imPath']), tid)
#                 writer.add_scalar('training/gpus', opt.num_gpus, tid)

#         # if opt.is_master and tid % 100 == 0:
#         #     for im_index, (im_single, semseg_color) in enumerate(zip(data_batch['im_not_hdr'], output_dict['semseg_color_list'])):
#         #         im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
#         #         im_path = os.path.join('./tmp/', 'im_%d-%d_color.png'%(tid, im_index))
#         #         color_path = os.path.join('./tmp/', 'im_%d-%d_semseg.png'%(tid, im_index))
#         #         cv2.imwrite(im_path, im_single * 255.)
#         #         semseg_color.save(color_path)

#         tid += 1



#             # print(ts_iter_end_start_list, ts_iter_start_end_list)