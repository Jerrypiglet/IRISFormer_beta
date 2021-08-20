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
from dataset_openrooms_OR_scanNetPose_binary_tables_ import openrooms_binary
from dataset_openrooms_OR_scanNetPose_pickle import openrooms_pickle
from utils.utils_dataloader_binary import make_data_loader_binary
import torch.distributed as dist
from train_funcs_detectron import gather_lists


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

# >>>>>>>>>>>>> DETECTRON setups
# import detectron2
# from detectron2.utils.logger import setup_logger
# from detectron2 import model_zoo
# from detectron2.structures import BoxMode
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer,ColorMode
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# import os.path as osp
# from detectron2.engine import DefaultTrainer
# from detectron2.utils.events import EventStorage

# cfg_detectron = get_cfg()

# # detectron_configs = utils_config.load_cfg_from_cfg_file(os.path.join(pwdpath, opt.cfg.MODEL_detectron.config_file))
# cfg_detectron.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
# # cfg_detectron.DATASETS.TRAIN = ("light_train",)
# # cfg_detectron.DATASETS.TEST = ("light_test", )
# # cfg_detectron.DATALOADER.NUM_WORKERS = 2
# cfg_detectron.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg_detectron.SOLVER.IMS_PER_BATCH = 2
# # cfg_detectron.SOLVER.BASE_LR = 0.001
# # cfg_detectron.SOLVER.MAX_ITER = args.iter    
# cfg_detectron.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
# # if(args.detect_4):
# cfg_detectron.MODEL.ROI_HEADS.NUM_CLASSES = 45 # OR45: 45 (-1 for unlabelled)
# # cfg_detectron.MODEL.TENSOR_MASK.MASK_LOSS_WEIGHT = 3
# # cfg_detectron.SOLVER.WEIGHT_DECAY: 0.0001
# cfg_detectron.MODEL.ROI_HEADS.NMS_THRESH_TEST: 0.7
# # cfg_detectron.SOLVER.MOMENTUM: 0.9
# cfg_detectron.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
# # cfg_detectron.OUTPUT_DIR=outfiledir+'model_path/'
# # False to include all empty image
# cfg_detectron.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
# cfg_detectron.INPUT.FORMAT = "RGB"

# cfg_detectron = utils_config.merge_cfg_from_list(cfg_detectron, opt.params)
# opt.cfg_detectron = cfg_detectron

# <<<<<<<<<<<<< DETECTRON setups

# >>>>>>>>>>>>> MODEL AND OPTIMIZER
from models_def.model_joint_all import Model_Joint as the_model
# build model
# model = MatSeg_BRDF(opt, logger)
model = the_model(opt, logger)
if opt.distributed: # https://github.com/dougsouza/pytorch-sync-batchnorm-example # export NCCL_LL_THRESHOLD=0
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(opt.device)
if opt.cfg.MODEL_BRDF.load_pretrained_pth:
    model.load_pretrained_MODEL_BRDF(opt.cfg.MODEL_BRDF.weights)
if opt.cfg.MODEL_SEMSEG.enable and opt.cfg.MODEL_SEMSEG.if_freeze:
    # model.turn_off_names(['UNet'])
    model.turn_off_names(['SEMSEG_Net'])
    model.freeze_bn_semantics()
if opt.cfg.MODEL_MATSEG.enable and opt.cfg.MODEL_MATSEG.if_freeze:
    model.turn_off_names(['MATSEG_Net'])
    model.freeze_bn_matseg()

model.print_net()

# print('+++', model.BRDF_Net.pretrained.model.patch_embed.backbone.stem.norm.weight)


# set up optimizers
# optimizer = get_optimizer(model.parameters(), cfg.SOLVER)
optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.lr, betas=(0.5, 0.999) )
if 'dpt_hybrid' in opt.cfg.MODEL_BRDF.DPT_baseline.model and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.dual_lr:
    backbone_params = []
    other_params = []
    for k, v in model.named_parameters():
        if 'BRDF_Net.pretrained.model.patch_embed.backbone' in k:
            backbone_params.append(v)
        else:
            other_params.append(v)
    # my_list = ['BRDF_Net.pretrained.model.patch_embed.backbone']
    # backbone_params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
    # other_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))
    optimizer_backbone = optim.Adam(backbone_params, lr=1e-5, betas=(0.5, 0.999) )
    optimizer_others = optim.Adam(other_params, lr=1e-5, betas=(0.5, 0.999) )

if opt.cfg.MODEL_BRDF.DPT_baseline.enable and opt.cfg.MODEL_BRDF.DPT_baseline.if_SGD:
    assert False, 'SGD disabled.'
    # optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.lr, momentum=0.9)
   

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

opt.OR_classes = OR4XCLASSES_dict[opt.cfg.MODEL_LAYOUT_EMITTER.data.OR][1:] # list of OR class names exclusing unlabelled(0)


# >>>>>>>>>>>>> DATASET
from utils.utils_semseg import get_transform_semseg, get_transform_matseg, get_transform_resize

transforms_train_semseg = get_transform_semseg('train', opt)
transforms_val_semseg = get_transform_semseg('val', opt)
transforms_train_matseg = get_transform_matseg('train', opt)
transforms_val_matseg = get_transform_matseg('val', opt)
transforms_train_resize = get_transform_resize('train', opt)
transforms_val_resize = get_transform_resize('val', opt)

openrooms_to_use = openrooms
make_data_loader_to_use = make_data_loader

if opt.cfg.DATASET.if_pickle:
    openrooms_to_use = openrooms_pickle
if opt.cfg.DATASET.if_binary:
    assert False, 'not supporting image resizing'
    openrooms_to_use = openrooms_binary
    make_data_loader_to_use = make_data_loader_binary
    
print('+++++++++openrooms_to_use', openrooms_to_use)

if opt.if_train:
    brdf_dataset_train = openrooms_to_use(opt, 
        transforms_fixed = transforms_val_resize, 
        transforms_semseg = transforms_train_semseg, 
        transforms_matseg = transforms_train_matseg,
        transforms_resize = transforms_train_resize, 
        cascadeLevel = opt.cascadeLevel, split = 'train', if_for_training=True, logger=logger)
    brdf_loader_train, _ = make_data_loader_to_use(
        opt,
        brdf_dataset_train,
        is_train=True,
        start_iter=0,
        logger=logger,
        collate_fn=collate_fn_OR, 
        # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
)
if opt.cfg.MODEL_SEMSEG.enable:
    opt.semseg_colors = brdf_dataset_train.semseg_colors

# if 'mini' in opt.cfg.DATASET.dataset_path:
#     print('=====!!!!===== mini: brdf_dataset_val = brdf_dataset_train')
#     brdf_dataset_val = brdf_dataset_train
#     brdf_dataset_val_vis = brdf_dataset_train
# else:

if opt.if_val:
    brdf_dataset_val = openrooms_to_use(opt, 
        transforms_fixed = transforms_val_resize, 
        transforms_semseg = transforms_val_semseg, 
        transforms_matseg = transforms_val_matseg,
        transforms_resize = transforms_val_resize, 
        # cascadeLevel = opt.cascadeLevel, split = 'val', logger=logger)
        # cascadeLevel = opt.cascadeLevel, split = 'val', load_first = 20 if opt.mini_val else -1, logger=logger)
        cascadeLevel = opt.cascadeLevel, split = 'val', if_for_training=False, load_first = -1, logger=logger)
    brdf_loader_val, _ = make_data_loader_to_use(
        opt,
        brdf_dataset_val,
        is_train=False,
        start_iter=0,
        logger=logger,
        # pin_memory = False, 
        collate_fn=collate_fn_OR, 
        # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
        if_distributed_override=opt.cfg.DATASET.if_val_dist and opt.distributed # default: True; -> should use gather from all GPUs if need all batches
    )

if opt.if_overfit_val and opt.if_train:
    brdf_dataset_train = openrooms_to_use(opt, 
        transforms_fixed = transforms_val_resize, 
        transforms_semseg = transforms_val_semseg, 
        transforms_matseg = transforms_val_matseg,
        transforms_resize = transforms_val_resize, 
        cascadeLevel = opt.cascadeLevel, split = 'val', if_for_training=True, load_first = -1, logger=logger)

    brdf_loader_train, _ = make_data_loader_to_use(
        opt,
        brdf_dataset_train,
        is_train=True,
        start_iter=0,
        logger=logger,
        collate_fn=collate_fn_OR, 
        # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
    )

if opt.if_overfit_train and opt.if_val:
    brdf_dataset_val = openrooms_to_use(opt, 
        transforms_fixed = transforms_val_resize, 
        transforms_semseg = transforms_val_semseg, 
        transforms_matseg = transforms_val_matseg,
        transforms_resize = transforms_val_resize, 
        # cascadeLevel = opt.cascadeLevel, split = 'val', logger=logger)
        # cascadeLevel = opt.cascadeLevel, split = 'val', load_first = 20 if opt.mini_val else -1, logger=logger)
        cascadeLevel = opt.cascadeLevel, split = 'train', if_for_training=False, load_first = -1, logger=logger)
    brdf_loader_val, _ = make_data_loader_to_use(
        opt,
        brdf_dataset_val,
        is_train=False,
        start_iter=0,
        logger=logger,
        # pin_memory = False, 
        collate_fn=collate_fn_OR, 
        # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
        if_distributed_override=opt.cfg.DATASET.if_val_dist and opt.distributed # default: True; -> should use gather from all GPUs if need all batches
    )

if opt.if_vis:
    brdf_dataset_val_vis = openrooms(opt, 
        transforms_fixed = transforms_val_resize, 
        transforms_semseg = transforms_val_semseg, 
        transforms_matseg = transforms_val_matseg,
        transforms_resize = transforms_val_resize, 
        cascadeLevel = opt.cascadeLevel, split = 'val', task='vis', if_for_training=False, load_first = opt.cfg.TEST.vis_max_samples, logger=logger)
    brdf_loader_val_vis, batch_size_val_vis = make_data_loader(
        opt,
        brdf_dataset_val_vis,
        is_train=False,
        start_iter=0,
        logger=logger,
        workers=2,
        batch_size_override=opt.batch_size_override_vis, 
        # pin_memory = False, 
        collate_fn=collate_fn_OR, 
        # collate_fn=my_collate_seq_dataset if opt.if_padding else my_collate_seq_dataset_noPadding,
        if_distributed_override=False
    )
    if opt.if_overfit_train:
        brdf_dataset_val_vis = openrooms(opt, 
            transforms_fixed = transforms_val_resize, 
            transforms_semseg = transforms_val_semseg, 
            transforms_matseg = transforms_val_matseg,
            transforms_resize = transforms_val_resize, 
            cascadeLevel = opt.cascadeLevel, split = 'train', task='vis', if_for_training=False, load_first = opt.cfg.TEST.vis_max_samples, logger=logger)
        brdf_loader_val_vis, batch_size_val_vis = make_data_loader(
            opt,
            brdf_dataset_val_vis,
            is_train=False,
            start_iter=0,
            logger=logger,
            workers=2,
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
from train_funcs_joint_all import get_labels_dict_joint, val_epoch_joint, vis_val_epoch_joint, forward_joint, get_time_meters_joint
from train_funcs_dump import dump_joint

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

if opt.distributed and 'hybrid' in opt.cfg.MODEL_BRDF.DPT_baseline.model and opt.cfg.MODEL_BRDF.DPT_baseline.enable:
    print(model.module.BRDF_Net.pretrained.model.patch_embed.backbone.stem.norm.bias)
else:
    print(model.BRDF_Net.pretrained.model.patch_embed.backbone.stem.norm.bias)
'''
tensor([ 7.6822e-02,  1.3747e-01,  1.7786e-01,  1.2228e-01,  1.3169e-01,
        -1.9725e-05,  1.8636e-01,  1.1685e-01,  1.2985e-01,  1.1506e-01,
         1.0491e-01,  1.4802e-01,  1.5881e-01,  2.9753e-01,  1.1565e-01,
         1.1791e-01,  1.2548e-01,  2.7642e-01,  1.1524e-01,  1.1916e-01,
         1.2935e-01,  3.2841e-01,  9.7098e-02,  1.2557e-01,  1.1197e-01,
         1.2638e-01,  1.5132e-01,  2.7703e-01,  1.3556e-01,  1.9152e-01,
         9.4249e-02,  1.4314e-01,  1.1369e-01,  1.3198e-01,  1.3196e-01,
         1.9317e-01,  1.2727e-01,  1.1400e-01,  1.1572e-01,  1.1789e-01,
         1.1287e-01,  9.6660e-02,  1.0051e-01,  1.4812e-01,  1.8120e-01,
         2.9408e-01,  8.3459e-02,  1.4994e-01,  2.5048e-01,  1.8411e-01,
         1.1894e-01,  2.1533e-01, -4.0096e-04,  1.3774e-01,  1.4944e-01,
         1.3586e-01,  1.3746e-01,  1.5885e-01,  1.2834e-01,  1.3462e-01,
         1.7779e-01,  1.2519e-01,  2.4257e-01,  1.7199e-01], device='cuda:0',
'''

# for epoch in list(range(opt.epochIdFineTune+1, opt.cfg.SOLVER.max_epoch)):
# for epoch_0 in list(range(1, 2) ):

if opt.cfg.MODEL_LAYOUT_EMITTER.mesh_obj.log_valid_objs:
    assert opt.if_cluster == False
    obj_nums_dict_file = Path(opt.cfg.DATASET.dataset_list) / 'obj_nums_dict.pickle' # {frame_info: (valid) obj_nums}
    import pickle
    if obj_nums_dict_file.exists():
        with open(obj_nums_dict_file, 'rb') as f:
            train_obj_dict = pickle.load(f)
            print(yellow('Existing keys in obj_nums_dict.pickle (%s):'%(str(obj_nums_dict_file))))
            for key in train_obj_dict:
                ic(key)
    else:
        print(yellow('Empty obj_nums_dict.pickle (%s)'%(str(obj_nums_dict_file))))
        train_obj_dict = {}

if not opt.if_train:
    val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid, 'bin_mean_shift': bin_mean_shift, 'if_register_detectron_only': False}
    if opt.if_vis:
        val_params.update({'batch_size_val_vis': batch_size_val_vis, 'detectron_dataset_name': 'vis'})
        with torch.no_grad():
            vis_val_epoch_joint(brdf_loader_val_vis, model, val_params)
        synchronize()
    if opt.if_val:
        val_params.update({'detectron_dataset_name': 'val'})
        with torch.no_grad():
            val_epoch_joint(brdf_loader_val, model, val_params)
else:
    for epoch_0 in list(range(opt.cfg.SOLVER.max_epoch)):
        epoch = epoch_0 + epoch_start

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
        
        start_iter = tid_start + len(brdf_loader_train) * epoch_0
        logger.info("Starting training from iteration {}".format(start_iter))
        # with EventStorage(start_iter) as storage:

        if cfg.SOLVER.if_test_dataloader:
            tic = time.time()
            tic_list = []

        count_samples_this_rank = 0

        for i, data_batch in tqdm(enumerate(brdf_loader_train)):

            if opt.cfg.DATASET.if_binary and opt.distributed:
                count_samples_this_rank += len(data_batch['frame_info'])
                count_samples_gathered = gather_lists([count_samples_this_rank], opt.num_gpus)
                # print('->', i, opt.rank)
                if opt.rank==0:
                    print('-', count_samples_gathered, '-', len(brdf_dataset_train.scene_key_frame_id_list_this_rank))
            
                if max(count_samples_gathered)>=len(brdf_dataset_train.scene_key_frame_id_list_this_rank):
                    break


            if cfg.SOLVER.if_test_dataloader:
                if i % 100 == 0:
                    print(data_batch.keys())
                    print(opt.task_name, 'On average: %.4f iter/s'%((len(tic_list)+1e-6)/(sum(tic_list)+1e-6)))
                if opt.cfg.MODEL_LAYOUT_EMITTER.mesh_obj.log_valid_objs:
                    # print(data_batch.keys())
                    # print(data_batch['boxes_valid_list'])
                    # print(data_batch['frame_info'])
                    ic(data_batch['num_valid_boxes'])
                    frame_info_list = data_batch['frame_info']
                    boxes_valid_list_list = data_batch['boxes_valid_list']
                    for frame_info, boxes_valid_list in zip(frame_info_list, boxes_valid_list_list):
                        frame_key = '%s-%s-%d'%(frame_info['meta_split'], frame_info['scene_name'], frame_info['frame_id'])
                        if frame_key in train_obj_dict:
                            continue
                        else:
                            if sum(boxes_valid_list) == 0:
                                print(sum(boxes_valid_list), boxes_valid_list)
                            train_obj_dict[frame_key] = {'valid_obj_num': sum(boxes_valid_list), 'boxes_valid_list': boxes_valid_list}
                tic_list.append(time.time()-tic)
                tic = time.time()
                continue
            reset_tictoc = False
            # Evaluation for an epoch```

            # synchronize()
            print((tid - tid_start) % opt.eval_every_iter, opt.eval_every_iter)
            if opt.eval_every_iter != -1 and (tid - tid_start) % opt.eval_every_iter == 0:
                val_params = {'writer': writer, 'logger': logger, 'opt': opt, 'tid': tid, 'bin_mean_shift': bin_mean_shift, 'if_register_detectron_only': False}
                if opt.if_vis:
                    val_params.update({'batch_size_val_vis': batch_size_val_vis, 'detectron_dataset_name': 'vis'})
                    with torch.no_grad():
                        if opt.cfg.DEBUG.if_dump_anything:
                            dump_joint(brdf_loader_val_vis, model, val_params)
                        vis_val_epoch_joint(brdf_loader_val_vis, model, val_params)
                    synchronize()                
                if opt.if_val:
                    val_params.update({'brdf_dataset_val': brdf_dataset_val, 'detectron_dataset_name': 'val'})
                    with torch.no_grad():
                        val_epoch_joint(brdf_loader_val, model, val_params)
                model.train(not cfg.MODEL_SEMSEG.fix_bn)
                reset_tictoc = True
                
                synchronize()

            # Save checkpoint
            if opt.save_every_iter != -1 and (tid - tid_start) % opt.save_every_iter == 0 and 'tmp' not in opt.task_name:
                check_save(opt, tid, tid, epoch, checkpointer, epochs_saved, opt.checkpoints_path_task, logger)
                reset_tictoc = True

            # synchronize()

            # torch.cuda.empty_cache()

            if reset_tictoc:
                ts_iter_end = time.time()
            ts_iter_start = time.time()
            if tid > 5:
                ts_iter_end_start_list.append(ts_iter_start - ts_iter_end)

            # if opt.ifDataloaderOnly:
            #     continue
            if tid % opt.debug_every_iter == 0:
                opt.if_vis_debug_pac = True


            # ======= Load data from cpu to gpu

            labels_dict = get_labels_dict_joint(data_batch, opt)
            # synchronize()

            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            if 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list or 'mesh' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                print('Valid objs num: ', [sum(x) for x in data_batch['boxes_valid_list']], 'Totasl objs num: ', [len(x) for x in data_batch['boxes_valid_list']])

            # ======= Forward
            if 'dpt_hybrid' in opt.cfg.MODEL_BRDF.DPT_baseline.model and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.dual_lr:
                optimizer_backbone.zero_grad()
                optimizer_others.zero_grad()
            else:
                optimizer.zero_grad()
            output_dict, loss_dict = forward_joint(True, labels_dict, model, opt, time_meters, tid=tid)
            # synchronize()
            
            # print('=======loss_dict', loss_dict)
            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()

            # ======= Backward
            loss = 0.
            loss_keys_backward = []
            loss_keys_print = []
            if opt.cfg.MODEL_MATSEG.enable and (not opt.cfg.MODEL_MATSEG.if_freeze):
                #  and ((not opt.cfg.MODEL_MATSEG.freeze) or opt.cfg.MODEL_MATSEG.embed_dims <= 4):
                loss_keys_backward.append('loss_matseg-ALL')
                loss_keys_print.append('loss_matseg-ALL')
                loss_keys_print.append('loss_matseg-pull')
                loss_keys_print.append('loss_matseg-push')
                loss_keys_print.append('loss_matseg-binary')

            if (opt.cfg.MODEL_SEMSEG.enable and not opt.cfg.MODEL_SEMSEG.if_freeze) or (opt.cfg.MODEL_BRDF.enable_semseg_decoder):
                loss_keys_backward.append('loss_semseg-ALL')
                loss_keys_print.append('loss_semseg-ALL')
                if opt.cfg.MODEL_SEMSEG.enable:
                    loss_keys_print.append('loss_semseg-main') 
                    loss_keys_print.append('loss_semseg-aux') 

            if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
                if not opt.cfg.MODEL_BRDF.if_freeze:
                    loss_keys_backward.append('loss_brdf-ALL')
                    loss_keys_print.append('loss_brdf-ALL')
                if 'al' in opt.cfg.MODEL_BRDF.enable_list and 'al' in opt.cfg.MODEL_BRDF.loss_list:
                    loss_keys_print.append('loss_brdf-albedo') 
                    if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_albedo:
                        loss_keys_print.append('loss_brdf-albedo-reg') 
                if 'no' in opt.cfg.MODEL_BRDF.enable_list and 'no' in opt.cfg.MODEL_BRDF.loss_list:
                    loss_keys_print.append('loss_brdf-normal') 
                if 'ro' in opt.cfg.MODEL_BRDF.enable_list and 'ro' in opt.cfg.MODEL_BRDF.loss_list:
                    loss_keys_print.append('loss_brdf-rough') 
                if 'de' in opt.cfg.MODEL_BRDF.enable_list and 'de' in opt.cfg.MODEL_BRDF.loss_list:
                    loss_keys_print.append('loss_brdf-depth') 
                    if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_depth:
                        loss_keys_print.append('loss_brdf-depth-reg') 

            if opt.cfg.MODEL_LIGHT.enable:
                if not opt.cfg.MODEL_LIGHT.if_freeze:
                    loss_keys_backward.append('loss_light-ALL')
                    loss_keys_print.append('loss_light-ALL')

            if opt.cfg.MODEL_MATCLS.enable:
                loss_keys_backward.append('loss_matcls-ALL')
                loss_keys_print.append('loss_matcls-ALL')
                loss_keys_print.append('loss_matcls-cls')
                if opt.cfg.MODEL_MATCLS.if_est_sup:
                    loss_keys_print.append('loss_matcls-supcls')


            if opt.cfg.MODEL_LAYOUT_EMITTER.enable:
                if_use_layout_loss = 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list and 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.loss_list
                if_use_object_loss = 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list and 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.loss_list

                if if_use_layout_loss:
                    if not opt.cfg.MODEL_LAYOUT_EMITTER.layout.if_freeze:
                        loss_keys_backward.append('loss_layout-ALL')
                    loss_keys_print.append('loss_layout-ALL')
                    loss_keys_print += ['loss_layout-pitch_cls', 
                        'loss_layout-pitch_reg', 
                        'loss_layout-roll_cls', 
                        'loss_layout-roll_reg', 
                        'loss_layout-lo_ori_cls', 
                        'loss_layout-lo_ori_reg', 
                        'loss_layout-lo_centroid', 
                        'loss_layout-lo_coeffs', 
                        'loss_layout-lo_corner', ]

                if if_use_object_loss:
                    loss_keys_backward.append('loss_object-ALL')
                    loss_keys_print.append('loss_object-ALL')
                if if_use_layout_loss and if_use_object_loss:
                    loss_keys_backward.append('loss_joint-ALL')
                    loss_keys_print.append('loss_joint-ALL')

                if 'mesh' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list and 'mesh' in opt.cfg.MODEL_LAYOUT_EMITTER.loss_list:
                    loss_keys_backward.append('loss_mesh-ALL')
                    loss_keys_print.append('loss_mesh-ALL')
                    if opt.cfg.MODEL_LAYOUT_EMITTER.mesh.loss == 'SVRLoss':
                        for loss_name in ['loss_mesh-chamfer', 'loss_mesh-face', 'loss_mesh-edge', 'loss_mesh-boundary']:
                            loss_keys_print.append(loss_name)
                        

                if 'em' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list and 'em' in opt.cfg.MODEL_LAYOUT_EMITTER.loss_list:
                    if not opt.cfg.MODEL_LAYOUT_EMITTER.emitter.if_freeze:
                        loss_keys_backward.append('loss_emitter-ALL')
                    loss_keys_print.append('loss_emitter-ALL')
                    loss_keys_print += ['loss_emitter-light_ratio', 
                        'loss_emitter-cell_cls', 
                        'loss_emitter-cell_axis', 
                        'loss_emitter-cell_intensity', 
                        'loss_emitter-cell_lamb'] 

            if opt.cfg.MODEL_DETECTRON.enable:
                loss_keys_backward.append('loss_detectron-ALL')
                loss_keys_print += ['loss_detectron-ALL', 
                    'loss_detectron-cls', 
                    'loss_detectron-box_reg', 
                    'loss_detectron-mask', 
                    'loss_detectron-rpn_cls', 
                    'loss_detectron-rpn_loc', ]

            for loss_key in loss_keys_backward:
                if loss_key in opt.loss_weight_dict:
                    loss_dict[loss_key] = loss_dict[loss_key] * opt.loss_weight_dict[loss_key]
                    print('Multiply loss %s by weight %.3f'%(loss_key, opt.loss_weight_dict[loss_key]))
            loss = sum([loss_dict[loss_key] for loss_key in loss_keys_backward])

            if opt.is_master and tid % 20 == 0:
                print('----loss_dict', loss_dict.keys())
                print('----loss_keys_backward', loss_keys_backward)

            loss.backward()

            # clip_to = 1.
            # torch.nn.utils.clip_grad_norm_(model.LAYOUT_EMITTER_NET_fc.parameters(), clip_to)
            # torch.nn.utils.clip_grad_norm_(model.LAYOUT_EMITTER_NET_encoder.parameters(), clip_to)
            # print(model.LAYOUT_EMITTER_NET_fc.fc_layout_5.weight)
            # print(model.LAYOUT_EMITTER_NET_fc.fc_layout_5.weight.grad)

            if 'dpt_hybrid' in opt.cfg.MODEL_BRDF.DPT_baseline.model and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.dual_lr:
                optimizer_backbone.step()
                optimizer_others.step()
            else:
                optimizer.step()
            time_meters['backward'].update(time.time() - time_meters['ts'])
            time_meters['ts'] = time.time()
            # synchronize()

            if opt.is_master:
                loss_keys_print = [x for x in loss_keys_print if 'ALL' in x] + [x for x in loss_keys_print if 'ALL' not in x]
                logger_str = 'Epoch %d - Tid %d -'%(epoch, tid) + ', '.join(['%s %.3f'%(loss_key, loss_dict_reduced[loss_key]) for loss_key in loss_keys_print])
                logger.info(white_blue(logger_str))

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
                # if len(ts_iter_end_start_list) > 100:
                #     ts_iter_end_start_list = []
                #     ts_iter_start_end_list = []
                if opt.is_master and tid % 100 == 0:
                    usage_ratio = print_gpu_usage(handle, logger)
                    writer.add_scalar('training/GPU_usage_ratio', usage_ratio, tid)
                    writer.add_scalar('training/batch_size_per_gpu', len(data_batch['image_path']), tid)
                    writer.add_scalar('training/gpus', opt.num_gpus, tid)
                    writer.add_scalar('training/lr', optimizer.param_groups[0]['lr'], tid)
            # if opt.is_master:

            # if tid % opt.debug_every_iter == 0:       
            #     if (opt.cfg.MODEL_MATSEG.if_albedo_pooling or opt.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv or opt.cfg.MODEL_MATSEG.if_albedo_pac_pool or opt.cfg.MODEL_MATSEG.if_albedo_safenet) and opt.cfg.MODEL_MATSEG.albedo_pooling_debug:
            #         if opt.is_master and output_dict['im_trainval_RGB_mask_pooled_mean'] is not None:
            #             for sample_idx, im_trainval_RGB_mask_pooled_mean in enumerate(output_dict['im_trainval_RGB_mask_pooled_mean']):
            #                 im_trainval_RGB_mask_pooled_mean = im_trainval_RGB_mask_pooled_mean.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            #                 writer.add_image('TRAIN_im_trainval_RGB_debug/%d'%(sample_idx+(tid*opt.cfg.SOLVER.ims_per_batch)), data_batch['im_trainval_RGB'][sample_idx].numpy().squeeze().transpose(1, 2, 0), tid, dataformats='HWC')
            #                 writer.add_image('TRAIN_im_trainval_RGB_mask_pooled_mean/%d'%(sample_idx+(tid*opt.cfg.SOLVER.ims_per_batch)), im_trainval_RGB_mask_pooled_mean, tid, dataformats='HWC')
            #                 logger.info('Added debug pooling sample')
            
            # ===== Logging summaries of training samples
            # if tid % 2000 == 0:
            #     for sample_idx, (im_single, im_trainval_RGB, im_path) in enumerate(zip(data_batch['im_trainval'], data_batch['im_trainval_RGB'], data_batch['image_path'])):
            #         # im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
            #         im_trainval_RGB = im_trainval_RGB.numpy().squeeze().transpose(1, 2, 0)
            #         if opt.is_master:
            #             # writer.add_image('TRAIN_im_trainval/%d'%sample_idx, im_single, tid, dataformats='HWC')
            #             writer.add_image('TRAIN_im_trainval_RGB/%d'%sample_idx, im_trainval_RGB, tid, dataformats='HWC')
            #             writer.add_text('TRAIN_image_name/%d'%sample_idx, im_path, tid)
            #     if opt.cfg.DATA.load_matseg_gt:
            #         for sample_idx, (im_single, mat_aggre_map) in enumerate(zip(data_batch['im_matseg_transformed_trainval'], labels_dict['mat_aggre_map_cpu'])):
            #             im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
            #             mat_aggre_map = mat_aggre_map.numpy().squeeze()
            #             if opt.is_master:
            #                 writer.add_image('TRAIN_matseg_im_trainval/%d'%sample_idx, im_single, tid, dataformats='HWC')
            #                 writer.add_image('TRAIN_matseg_mat_aggre_map_trainval/%d'%sample_idx, vis_index_map(mat_aggre_map), tid, dataformats='HWC')
            #             logger.info('Logged training mat seg')
            #     if opt.cfg.DATA.load_semseg_gt and opt.cfg.MODEL_SEMSEG.enable:
            #         for sample_idx, (im_single, semseg_label, semseg_pred) in enumerate(zip(data_batch['im_semseg_transformed_trainval'], data_batch['semseg_label'], output_dict['semseg_pred'].detach().cpu())):
            #             im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
            #             semseg_colors = np.loadtxt(os.path.join(opt.pwdpath, opt.cfg.PATH.semseg_colors_path)).astype('uint8')
            #             if opt.cfg.MODEL_SEMSEG.wallseg_only:
            #                 semseg_colors = np.array([[0, 0, 0], [0, 80, 100]], dtype=np.uint8)

            #             semseg_label = np.uint8(semseg_label.numpy().squeeze())
            #             from utils.utils_vis import colorize
            #             semseg_label_color = np.array(colorize(semseg_label, semseg_colors).convert('RGB'))
            #             if opt.is_master:
            #                 writer.add_image('TRAIN_semseg_im_trainval/%d'%sample_idx, im_single, tid, dataformats='HWC')
            #                 writer.add_image('TRAIN_semseg_label_trainval/%d'%sample_idx, semseg_label_color, tid, dataformats='HWC')

            #             prediction = np.argmax(semseg_pred.numpy().squeeze(), 0)
            #             gray_pred = np.uint8(prediction)
            #             color_pred = np.array(colorize(gray_pred, semseg_colors).convert('RGB'))
            #             if opt.is_master:
            #                 writer.add_image('TRAIN_semseg_PRED/%d'%sample_idx, color_pred, tid, dataformats='HWC')

            #             logger.info('Logged training sem seg')

            # synchronize()
            if tid % opt.debug_every_iter == 0:
                opt.if_vis_debug_pac = False

            tid += 1
            if tid >= opt.max_iter and opt.max_iter != -1:
                break


if opt.cfg.MODEL_LAYOUT_EMITTER.mesh_obj.log_valid_objs:
    with open(obj_nums_dict_file, 'wb') as f:
        pickle.dump(train_obj_dict, f)
    print(white_blue('train_obj_dict dumped to %s'%obj_nums_dict_file))