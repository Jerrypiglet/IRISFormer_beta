import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from utils.utils_misc import *
from utils.comm import synchronize, get_rank
import os, sys
from utils.utils_total3D.data_config import Dataset_Config
from utils.utils_total3D.utils_OR_layout import to_dict_tensor


def set_up_envs(opt):
    opt.cfg.PATH.root = opt.cfg.PATH.root_cluster if opt.if_cluster else opt.cfg.PATH.root_local
    opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_cluster if opt.if_cluster else opt.cfg.DATASET.dataset_path_local
    opt.cfg.DATASET.layout_emitter_path = opt.cfg.DATASET.layout_emitter_path_cluster if opt.if_cluster else opt.cfg.DATASET.layout_emitter_path_local
    opt.cfg.DATASET.png_path = opt.cfg.DATASET.png_path_cluster if opt.if_cluster else opt.cfg.DATASET.png_path_local
    opt.cfg.DATASET.matpart_path = opt.cfg.DATASET.matpart_path_cluster if opt.if_cluster else opt.cfg.DATASET.matpart_path_local
    opt.cfg.DATASET.matori_path = opt.cfg.DATASET.matori_path_cluster if opt.if_cluster else opt.cfg.DATASET.matori_path_local
    if opt.data_root is not None:
        opt.cfg.DATASET.dataset_path = opt.data_root

    if opt.cfg.DATASET.mini:
        opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_mini
        opt.cfg.DATASET.dataset_list = opt.cfg.DATASET.dataset_list_mini

    opt.cfg.MODEL_SEMSEG.semseg_path = opt.cfg.MODEL_SEMSEG.semseg_path_cluster if opt.if_cluster else opt.cfg.MODEL_SEMSEG.semseg_path_local
    opt.cfg.PATH.semseg_colors_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.semseg_colors_path)
    opt.cfg.PATH.semseg_names_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.semseg_names_path)
    opt.cfg.PATH.total3D_colors_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.total3D_colors_path)
    opt.cfg.PATH.total3D_lists_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.total3D_lists_path)
    opt.cfg.PATH.OR4X_mapping_catInt_to_RGB = [os.path.join(opt.cfg.PATH.root, x) for x in opt.cfg.PATH.OR4X_mapping_catInt_to_RGB]
    opt.cfg.PATH.OR4X_mapping_catStr_to_RGB = [os.path.join(opt.cfg.PATH.root, x) for x in opt.cfg.PATH.OR4X_mapping_catStr_to_RGB]
    opt.cfg.PATH.matcls_matIdG1_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.matcls_matIdG1_path)
    opt.cfg.PATH.matcls_matIdG2_path = os.path.join(opt.cfg.PATH.root, opt.cfg.PATH.matcls_matIdG2_path)

    # ===== data =====
    opt.cfg.DATA.data_read_list = list(set(opt.cfg.DATA.data_read_list.split('_')))

    # ====== BRDF =====
    if len(opt.cfg.MODEL_BRDF.enable_list) > 0:
        opt.cfg.enable_BRDF_decoders = True
    if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
        opt.cfg.DATA.load_brdf_gt = True
        opt.depth_metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        opt.cfg.MODEL_BRDF.loss_list = opt.cfg.MODEL_BRDF.enable_list

    # ====== per-pixel lighting =====
    if opt.cfg.MODEL_LIGHT.enable:
        opt.cfg.DATA.load_light_gt = True
        opt.cfg.DATA.load_brdf_gt = True
        opt.cfg.DATA.data_read_list += ['al_no_de_ro'].split('_')
        if opt.cfg.MODEL_LIGHT.use_GT_brdf:
            opt.cfg.MODEL_BRDF.enable = False
            opt.cfg.MODEL_BRDF.enable_list = ''
        else:
            opt.cfg.MODEL_BRDF.enable = True
            opt.cfg.MODEL_BRDF.enable_list += 'al_no_de_ro'.split(['_'])
        opt.cfg.MODEL_BRDF.loss_list = ''

    # ====== layout, emitters =====
    if opt.cfg.MODEL_LAYOUT_EMITTER.enable:
        opt.cfg.MODEL_LAYOUT_EMITTER.enable = True
        opt.cfg.DATA.load_layout_emitter_gt = True
        opt.cfg.DATA.load_brdf_gt = True
        opt.cfg.DATA.data_read_list += opt.cfg.MODEL_LAYOUT_EMITTER.enable_list.split('_')
        if opt.cfg.MODEL_LAYOUT_EMITTER.use_depth_as_input:
            opt.cfg.DATA.data_read_list.append('de')
        assert opt.cfg.MODEL_LAYOUT_EMITTER.emitter.est_type in ['cell_prob', 'wall_prob', 'cell_info']

        opt.dataset_config = Dataset_Config('OR', OR=opt.cfg.MODEL_LAYOUT_EMITTER.data.OR, version=opt.cfg.MODEL_LAYOUT_EMITTER.data.version, opt=opt)
        opt.bins_tensor = to_dict_tensor(opt.dataset_config.bins, if_cuda=True)


    # ====== semseg =====
    if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable or opt.cfg.MODEL_SEMSEG.use_as_input or opt.cfg.MODEL_MATSEG.use_semseg_as_input:
        opt.cfg.DATA.load_semseg_gt = True
        opt.semseg_criterion = nn.CrossEntropyLoss(ignore_index=opt.cfg.DATA.semseg_ignore_label)
    
    if opt.cfg.MODEL_MATSEG.enable or opt.cfg.MODEL_MATSEG.use_as_input or opt.cfg.MODEL_MATSEG.if_albedo_pooling or opt.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv or opt.cfg.MODEL_MATSEG.if_albedo_pac_pool:
        opt.cfg.DATA.load_matseg_gt = True
    
    if opt.cfg.MODEL_BRDF.enable_semseg_decoder and opt.cfg.MODEL_SEMSEG.enable:
        raise (RuntimeError("Cannot be True at the same time: opt.cfg.MODEL_BRDF.enable_semseg_decoder, opt.cfg.MODEL_SEMSEG.enable"))

    if not opt.cfg.MODEL_SEMSEG.if_freeze:
        opt.cfg.MODEL_SEMSEG.fix_bn = False

    # ====== matcls =====
    if opt.cfg.MODEL_MATCLS.enable:
        opt.cfg.DATA.load_matcls_gt = True


    # ===== check if flags are legal =====
    check_if_in_list(opt.cfg.DATA.data_read_list, opt.cfg.DATA.data_read_list_allowed)

    opt.cfg.MODEL_BRDF.enable_list = opt.cfg.MODEL_BRDF.enable_list.split('_')
    opt.cfg.MODEL_BRDF.loss_list = opt.cfg.MODEL_BRDF.loss_list.split('_')
    check_if_in_list(opt.cfg.MODEL_BRDF.enable_list, opt.cfg.MODEL_BRDF.enable_list_allowed)
    check_if_in_list(opt.cfg.MODEL_BRDF.loss_list, opt.cfg.MODEL_BRDF.enable_list_allowed)

    opt.cfg.MODEL_LAYOUT_EMITTER.enable_list = opt.cfg.MODEL_LAYOUT_EMITTER.enable_list.split('_')
    if opt.cfg.MODEL_LAYOUT_EMITTER.loss_list == '':
        opt.cfg.MODEL_LAYOUT_EMITTER.loss_list = opt.cfg.MODEL_LAYOUT_EMITTER.enable_list
    else:
        opt.cfg.MODEL_LAYOUT_EMITTER.loss_list = opt.cfg.MODEL_LAYOUT_EMITTER.loss_list.split('_')
    check_if_in_list(opt.cfg.MODEL_LAYOUT_EMITTER.enable_list, opt.cfg.MODEL_LAYOUT_EMITTER.enable_list_allowed)
    check_if_in_list(opt.cfg.MODEL_LAYOUT_EMITTER.loss_list, opt.cfg.MODEL_LAYOUT_EMITTER.enable_list_allowed)

    # Guidance in general
    guidance_options = [opt.cfg.MODEL_MATSEG.if_albedo_pooling,opt.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv, \
        opt.cfg.MODEL_MATSEG.if_albedo_pac_pool, opt.cfg.MODEL_MATSEG.if_albedo_pac_conv, \
        opt.cfg.MODEL_MATSEG.if_albedo_safenet]
    from utils.utils_misc import only1true
    assert only1true(guidance_options) or nonetrue(guidance_options), 'Only ONE of the guidance methods canbe true at the same time!'

    assert opt.cfg.MODEL_MATSEG.albedo_pooling_from in ['gt', 'pred']

    # PAC
    opt.if_vis_debug_pac = False
    # Pac pool
    opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers  = opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers.split('_')
    opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers_allowed  = opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers_allowed.split('_')
    assert all(e in opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers_allowed for e in opt.cfg.MODEL_MATSEG.albedo_pac_pool_mean_layers)

    # Pac cov
    opt.cfg.MODEL_MATSEG.albedo_pac_conv_mean_layers  = opt.cfg.MODEL_MATSEG.albedo_pac_conv_mean_layers.split('_')
    opt.cfg.MODEL_MATSEG.albedo_pac_conv_mean_layers_allowed  = opt.cfg.MODEL_MATSEG.albedo_pac_conv_mean_layers_allowed.split('_')
    assert all(e in opt.cfg.MODEL_MATSEG.albedo_pac_conv_mean_layers_allowed for e in opt.cfg.MODEL_MATSEG.albedo_pac_conv_mean_layers)

    opt.cfg.MODEL_MATSEG.albedo_pac_conv_deform_layers  = opt.cfg.MODEL_MATSEG.albedo_pac_conv_deform_layers.split('_')
    opt.cfg.MODEL_MATSEG.albedo_pac_conv_deform_layers_allowed  = opt.cfg.MODEL_MATSEG.albedo_pac_conv_deform_layers_allowed.split('_')
    assert all(e in opt.cfg.MODEL_MATSEG.albedo_pac_conv_deform_layers_allowed for e in opt.cfg.MODEL_MATSEG.albedo_pac_conv_deform_layers)
    
    # Safenet global affinity
    opt.cfg.MODEL_MATSEG.albedo_safenet_affinity_layers  = opt.cfg.MODEL_MATSEG.albedo_safenet_affinity_layers.split('_')
    opt.cfg.MODEL_MATSEG.albedo_safenet_affinity_layers_allowed  = opt.cfg.MODEL_MATSEG.albedo_safenet_affinity_layers_allowed.split('_')
    assert all(e in opt.cfg.MODEL_MATSEG.albedo_safenet_affinity_layers_allowed for e in opt.cfg.MODEL_MATSEG.albedo_safenet_affinity_layers)

    # DCN
    opt.cfg.PATH.dcn_path = opt.cfg.PATH.dcn_cluster if opt.if_cluster else opt.cfg.PATH.dcn_local
    sys.path.insert(0, os.path.join(opt.cfg.PATH.dcn_path, 'functions'))

    # export
    opt.cfg.PATH.torch_home_path = opt.cfg.PATH.torch_home_cluster if opt.if_cluster else opt.cfg.PATH.torch_home_local
    os.system('export TORCH_HOME=%s'%opt.cfg.PATH.torch_home_path)

    # mis
    if opt.cfg.SOLVER.if_test_dataloader:
        opt.cfg.SOLVER.max_epoch = 1

def check_if_in_list(list_to_check, list_allowed, module_name='Unknown Module'):
    if len(list_to_check) == 0:
        return
    if isinstance(list_to_check, str):
        list_to_check = list_to_check.split('_')
    list_to_check = [x for x in list_to_check if x != '']
    if not all(e in list_allowed for e in list_to_check):
        print(list_to_check, list_allowed)
        error_str = red('Illegal %s of length %d: %s'%(module_name, len(list_to_check), '_'.join(list_to_check)))
        raise ValueError(error_str)



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
    # logger.info(red("==[opt.semseg_configs]=="))
    # logger.info(opt.semseg_configs)

    with open(opt.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        # logger.info(config_str)
    printer = printer(opt.rank, debug=opt.debug)

    if opt.is_master and 'tmp' not in opt.task_name:
        exclude_list = ['apex', 'logs_bkg', 'archive', 'train_cifar10_py', 'train_mnist_tf', 'utils_external', 'build/'] + \
            ['Summary', 'Summary_vis', 'Checkpoint', 'logs', '__pycache__', 'snapshots', '.vscode', '.ipynb_checkpoints', 'azureml-setup', 'azureml_compute_logs']
        if opt.if_cluster:
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
    #     opt.root = opt.cfg.PATH.root_local
    # else:
    #     opt.root = opt.cfg.PATH.root_cluster
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
        if_delete = 'n'
        print(green(opt.summary_path_task), os.path.isdir(opt.summary_path_task))
        if os.path.isdir(opt.summary_path_task):
            if 'POD' in opt.task_name:
                if_delete = 'n'
                opt.resume = opt.task_name
                print(green('Resuming task %s'%opt.resume))

                if opt.reset_latest_ckpt:
                    os.system('rm %s'%(os.path.join(opt.checkpoints_path_task, 'last_checkpoint')))
                    print(green('Removed last_checkpoint shortcut for %s'%opt.resume))

            #     if opt.resume == 'NoCkpt':
            #         if_delete = 'y'
            # else:
            #     if_delete = input(colored('Summary path %s already exists. Delete? [y/n] '%opt.summary_p[ath_task, 'white', 'on_blue'))
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
    opt.if_cuda = opt.device == 'cuda'
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
        print(opt.replaced_keys, opt.replacedby, )
        assert len(opt.replaced_keys) == len(opt.replacedby)
        replace_kws += opt.replaced_keys
        replace_with_kws += opt.replacedby
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