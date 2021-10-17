# import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils import data
import scipy.ndimage as ndimage
import cv2
from skimage.measure import block_reduce 
import h5py
import scipy.ndimage as ndimage
import torch
from tqdm import tqdm
import torchvision.transforms as T
# import PIL
from utils.utils_misc import *
from pathlib import Path
# import pickle
import pickle5 as pickle
from icecream import ic
from utils.utils_total3D.utils_OR_imageops import loadHdr_simple, to_nonhdr
import math
from utils.utils_total3D.data_config import RECON_3D_CLS_OR_dict
from scipy.spatial import cKDTree
import copy

from utils.utils_total3D.utils_OR_vis_labels import RGB_to_01
from utils.utils_total3D.utils_others import Relation_Config, OR4XCLASSES_dict, OR4XCLASSES_not_detect_mapping_ids_dict, OR4X_mapping_catInt_to_RGB

from utils.utils_scannet import read_ExtM_from_txt, read_img
import utils.utils_nvidia.mdataloader.m_preprocess as m_preprocess
import PIL
import torchvision.transforms as tfv_transform

import warnings
warnings.filterwarnings("ignore")

rel_cfg = Relation_Config()
d_model = int(rel_cfg.d_g/4)


HEIGHT_PATCH = 256
WIDTH_PATCH = 256

from utils import transform
def get_bdb2d_transform(split, crop_bdb): # crop_bdb: [x1, x2, y1, y2] in float
    assert split in ['train', 'val']
    if split == 'train':
        data_transforms_crop_nonormalize = [
            transform.CropBdb(crop_bdb), 
            transform.Resize((280, 280)),
            transform.Crop((HEIGHT_PATCH, WIDTH_PATCH), crop_type='rand'),
            transform.ToTensor()
        ]
        data_transforms_crop_nonormalize = transform.Compose(data_transforms_crop_nonormalize)
        return data_transforms_crop_nonormalize
    else:
        data_transforms_nocrop_nonormalize = [
            transform.CropBdb(crop_bdb), 
            transform.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
            transform.ToTensor()
        ]
        data_transforms_nocrop_nonormalize = transform.Compose(data_transforms_nocrop_nonormalize)
        return data_transforms_nocrop_nonormalize

def return_percent(list_in, percent=1.):
    len_list = len(list_in)
    return_len = max(1, int(np.floor(len_list*percent)))
    return list_in[:return_len]

def get_valid_scenes(opt, frames_list_path, split, logger=None):
    scenes_list_path = str(frames_list_path).replace('.txt', '_scenes.txt')
    if not os.path.isfile(scenes_list_path):
        raise (RuntimeError("Scene list file do not exist: " + scenes_list_path + "\n"))
    if logger is None:
        logger = basic_logger()

    meta_split_scene_name_list = []
    list_read = open(scenes_list_path).readlines()
    logger.info("Totally {} scenes in {} set.".format(len(list_read), split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            assert False, 'No support for test split for now.'
        else:
            if len(line_split) not in [2]:
                raise (RuntimeError("Scene list file read line error : " + line + "\n"))

        meta_split, scene_name = line_split
        meta_split_scene_name_list.append([meta_split, scene_name])

    return meta_split_scene_name_list

def make_dataset(opt, split, task, data_root=None, data_list=None, logger=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    if logger is None:
        logger = basic_logger()
    image_label_list = []
    meta_split_scene_name_frame_id_list = []
    list_read = open(data_list).readlines()
    logger.info("Totally {} samples in {} set.".format(len(list_read), split))
    logger.info("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            image_name = os.path.join(data_root, line_split[2])
            if len(line_split) != 3:
                label_name = os.path.join(data_root, line_split[3])
                # raise (RuntimeError("Image list file read line error : " + line + "\n"))
            else:
                label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) not in [3, 4]:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[2])
            # label_name = os.path.join(data_root, line_split[3])
            label_name = ''
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''

        meta_split = line_split[2].split('/')[0]
        # print(meta_split, opt.meta_splits_skip, meta_split in opt.meta_splits_skip)
        if opt.meta_splits_skip is not None and meta_split in opt.meta_splits_skip:
            continue
        item = (image_name, label_name)
        image_label_list.append(item)
        meta_split_scene_name_frame_id_list.append((meta_split, line_split[0], int(line_split[1])))

    logger.info("==> Checking image&label pair [%s] list done! %d frames."%(split, len(image_label_list)))

    all_scenes = get_valid_scenes(opt, data_list, split, logger=logger)

    if opt.cfg.DATASET.first_scenes != -1:
        # return image_label_list[:opt.cfg.DATASET.first_scenes], meta_split_scene_name_frame_id_list[:opt.cfg.DATASET.first_scenes]
        assert False
    # elif opt.cfg.DATASET.if_quarter and task != 'vis':
    elif opt.cfg.DATASET.if_quarter and task in ['train']:
        meta_split_scene_name_frame_id_list_quarter = return_percent(meta_split_scene_name_frame_id_list, 0.25)
        all_scenes = list(set(['/'.join([x[0], x[1]]) for x in meta_split_scene_name_frame_id_list_quarter]))
        all_scenes = [x.split('/') for x in all_scenes]
        return return_percent(image_label_list, 0.25), meta_split_scene_name_frame_id_list_quarter, all_scenes

        # all_scenes = return_percent(all_scenes, 0.25)
        # image_label_list_valid = []
        # meta_split_scene_name_frame_id_list_valid = []
        # for image_label, meta_split_scene_name_frame_id in zip(image_label_list, meta_split_scene_name_frame_id_list):
        #     meta_split, scene_name = meta_split_scene_name_frame_id[0], meta_split_scene_name_frame_id[1]
        #     if [meta_split, scene_name] in all_scenes:
        #         image_label_list_valid.append(image_label)
        #         meta_split_scene_name_frame_id_list_valid.append(meta_split_scene_name_frame_id)
        
        # return image_label_list_valid, meta_split_scene_name_frame_id_list_valid, all_scenes

    else:
        return image_label_list, meta_split_scene_name_frame_id_list, all_scenes