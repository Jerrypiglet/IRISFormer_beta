import os
import torch.nn as nn

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

    if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable or opt.cfg.MODEL_SEMSEG.use_as_input:
        opt.cfg.DATA.load_semseg_gt = True
        opt.semseg_criterion = nn.CrossEntropyLoss(ignore_index=opt.cfg.DATA.semseg_ignore_label)
    
    if opt.cfg.MODEL_BRDF.enable_semseg_decoder and opt.cfg.MODEL_SEMSEG.enable:
        raise (RuntimeError("Cannot be True at the same time: opt.cfg.MODEL_BRDF.enable_semseg_decoder, opt.cfg.MODEL_SEMSEG.enable"))

    if not opt.cfg.MODEL_SEMSEG.if_freeze:
        opt.cfg.MODEL_SEMSEG.fix_bn = False

