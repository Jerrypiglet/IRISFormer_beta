import os
import torch.nn as nn
import torch
import numpy as np
import random
from pathlib import Path
from utils.utils_misc import *
from utils.comm import synchronize, get_rank


def set_up_envs(opt):
    opt.cfg.PATH.root = opt.cfg.PATH.root_cluster if opt.if_cluster else opt.cfg.PATH.root_local
    opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_cluster if opt.if_cluster else opt.cfg.DATASET.dataset_path_local
    if opt.data_root is not None:
        opt.cfg.DATASET.dataset_path = opt.data_root
    opt.cfg.MODEL_SEMSEG.semseg_path = opt.cfg.MODEL_SEMSEG.semseg_path_cluster if opt.if_cluster else opt.cfg.MODEL_SEMSEG.semseg_path_local
    opt.cfg.PATH.semseg_colors_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.semseg_colors_path)
    opt.cfg.PATH.semseg_names_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.semseg_names_path)
    # for keys in ['PATH.semseg_colors_path', 'PATH.semseg_names_path']
    # keys_to_set_path = ['PATH.semseg_colors_path', 'PATH.semseg_names_path']
    # for cfg_key in opt.cfg:
    #     print(cfg_key)
        # for key_to_set_path in keys_to_set_path:
            # print(key_to_set_path, cfg_key)
            # if key_to_set_path in cfg_key:

    # opt.cfg.MODEL_MATSEG.matseg_path = opt.CKPT_PATH

    if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
        opt.cfg.DATA.load_brdf_gt = True
        opt.depth_metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']

    if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable or opt.cfg.MODEL_SEMSEG.use_as_input or opt.cfg.MODEL_MATSEG.use_semseg_as_input:
        opt.cfg.DATA.load_semseg_gt = True
        opt.semseg_criterion = nn.CrossEntropyLoss(ignore_index=opt.cfg.DATA.semseg_ignore_label)
    
    if opt.cfg.MODEL_MATSEG.enable or opt.cfg.MODEL_MATSEG.use_as_input:
        opt.cfg.DATA.load_matseg_gt = True
    
    if opt.cfg.MODEL_BRDF.enable_semseg_decoder and opt.cfg.MODEL_SEMSEG.enable:
        raise (RuntimeError("Cannot be True at the same time: opt.cfg.MODEL_BRDF.enable_semseg_decoder, opt.cfg.MODEL_SEMSEG.enable"))

    if not opt.cfg.MODEL_SEMSEG.if_freeze:
        opt.cfg.MODEL_SEMSEG.fix_bn = False

def set_up_logger(opt):
    from utils.logger import setup_logger, Logger, printer
    import sys
    from torch.utils.tensorboard import SummaryWriter

    # === LOGGING
    sys.stdout = Logger(Path(opt.summary_path_task) / 'log.txt')
    # sys.stdout = Logger(opt.summary_path_task / 'log.txt')
    logger = setup_logger("logger:train", opt.summary_path_task, opt.rank, filename="logger_maskrcn-style.txt")
    logger.info(red("==[config]== opt"))
    logger.info(opt)
    logger.info(red("==[config]== cfg"))
    logger.info(opt.cfg)
    logger.info(red("==[config]== Loaded configuration file {}".format(opt.config_file)))
    logger.info(red("==[opt.semseg_configs]=="))
    logger.info(opt.semseg_configs)

    with open(opt.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        # logger.info(config_str)
    printer = printer(opt.rank, debug=opt.debug)

    if opt.is_master and 'tmp' not in opt.task_name:
        exclude_list = ['apex', 'logs_bkg', 'archive', 'train_cifar10_py', 'train_mnist_tf', 'utils_external', 'build/'] + \
            ['Summary', 'Summary_vis', 'Checkpoint', 'logs', '__pycache__', 'snapshots', '.vscode', '.ipynb_checkpoints', 'azureml-setup', 'azureml_compute_logs']
        copy_py_files(opt.pwdpath, opt.summary_vis_path_task_py, exclude_paths=[str(opt.SUMMARY_PATH), str(opt.CKPT_PATH), str(opt.SUMMARY_VIS_PATH)]+exclude_list)
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

    return logger, writer



def set_up_folders(opt):
    from utils.global_paths import SUMMARY_PATH, SUMMARY_VIS_PATH, CKPT_PATH
    opt.SUMMARY_PATH, opt.SUMMARY_VIS_PATH, opt.CKPT_PATH = SUMMARY_PATH, SUMMARY_VIS_PATH, CKPT_PATH

    # >>>> SUMMARY WRITERS
    if opt.if_cluster:
        opt.home_path = Path('/viscompfs/users/ruizhu/')
        opt.CKPT_PATH = opt.home_path / CKPT_PATH
        opt.SUMMARY_PATH = opt.home_path / SUMMARY_PATH
        opt.SUMMARY_VIS_PATH = opt.home_path / SUMMARY_VIS_PATH
    if not opt.if_cluster:
        opt.task_name = get_datetime() + '-' + opt.task_name
        # print(opt.cfg)
        opt.root = opt.cfg.PATH.root_local
    else:
        opt.root = opt.cfg.PATH.root_cluster
    opt.summary_path_task = opt.SUMMARY_PATH / opt.task_name
    opt.checkpoints_path_task = opt.CKPT_PATH / opt.task_name
    opt.summary_vis_path_task = opt.SUMMARY_VIS_PATH / opt.task_name
    opt.summary_vis_path_task_py = opt.summary_vis_path_task / 'py_files'

    save_folders = [opt.summary_path_task, opt.summary_vis_path_task, opt.summary_vis_path_task_py, opt.checkpoints_path_task, ]
    print('====%d/%d'%(opt.rank, opt.num_gpus), opt.checkpoints_path_task)

    if opt.is_master:
        for root_folder in [opt.SUMMARY_PATH, opt.CKPT_PATH, opt.SUMMARY_VIS_PATH]:
            if not root_folder.exists():
                root_folder.mkdir(exist_ok=True)
        if os.path.isdir(opt.summary_path_task):
            if opt.task_name not in ['tmp'] and opt.resume == 'NoCkpt':
                if ('POD' in opt.task_name) and opt.resume != 'NoCkpt':
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


def set_up_dist(opt):
    import nvidia_smi

    # >>>> DISTRIBUTED TRAINING
    torch.manual_seed(opt.cfg.seed)
    np.random.seed(opt.cfg.seed)
    random.seed(opt.cfg.seed)

    opt.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.distributed = opt.num_gpus > 1
    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        process_group = torch.distributed.init_process_group(
            backend="nccl", world_size=opt.num_gpus, init_method="env://"
        )
        # synchronize()
    # device = torch.device("cuda" if torch.cuda.is_available() and not opt.cpu else "cpu")
    opt.device = 'cuda'
    opt.rank = get_rank()
    opt.is_master = opt.rank == 0
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(opt.rank)
    # <<<< DISTRIBUTED TRAINING
    return handle

def set_up_checkpointing(opt, model, optimizer, scheduler, logger):
    from utils.checkpointer import DetectronCheckpointer

    # >>>> CHECKPOINTING
    save_to_disk = opt.is_master
    checkpointer = DetectronCheckpointer(
        opt, model, optimizer, scheduler, opt.CKPT_PATH, opt.checkpoints_path_task, save_to_disk, logger=logger, if_reset_scheduler=opt.reset_scheduler
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

    if opt.reset_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.cfg.SOLVER.lr
            
    # <<<< CHECKPOINTING
    return checkpointer, tid_start, epoch_start