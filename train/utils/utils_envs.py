import os

def get_dataset_path(opt):
    if opt.if_cluster:
        opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_cluster
    else:
        opt.cfg.DATASET.dataset_path = opt.cfg.DATASET.dataset_path_local
    return opt.cfg.DATASET.dataset_path

def set_up_envs(opt):
    get_dataset_path(opt)

    opt.cfg.PATH.root = opt.cfg.PATH.root_cluster if opt.if_cluster else opt.cfg.PATH.root_local
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

