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

# import math

# HEIGHT_PATCH = 256
# WIDTH_PATCH = 256
from utils.utils_total3D.utils_OR_vis_labels import RGB_to_01
from utils.utils_total3D.utils_others import Relation_Config, OR4XCLASSES_dict, OR4XCLASSES_not_detect_mapping_ids_dict, OR4X_mapping_catInt_to_RGB
# OR = 'OR45'
# classes = OR4XCLASSES_dict[OR]


def make_dataset(split='train', data_root=None, data_list=None, logger=None):
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
            if len(line_split) != 4:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[2])
            label_name = os.path.join(data_root, line_split[3])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
        meta_split_scene_name_frame_id_list.append((line_split[2].split('/')[0], line_split[0], int(line_split[1])))
    logger.info("Checking image&label pair {} list done!".format(split))
    # print(image_label_list[:5])
    return image_label_list, meta_split_scene_name_frame_id_list


class openrooms(data.Dataset):
    def __init__(self, opt, data_list=None, logger=basic_logger(), transforms_fixed=None, transforms_semseg=None, transforms_matseg=None, transforms_resize=None, 
            split='train', load_first = -1, rseed = 1, 
            cascadeLevel = 0,
            # is_light = False, is_all_light = False,
            envHeight = 8, envWidth = 16, envRow = 120, envCol = 160, 
            SGNum = 12):

        if logger is None:
            logger = basic_logger()

        self.opt = opt
        self.cfg = self.opt.cfg
        self.logger = logger
        self.rseed = rseed
        self.dataset_name = 'scannet'
        self.split = 'test'
        assert self.split in ['train', 'val', 'test']

        self.data_list_file = self.opt.cfg.MODEL_MATCLS.real_images_list
        with open(self.data_list_file, 'r') as fIn:
            lines = [x.strip() for x in fIn.readlines()]
        self.data_list = [x.split(' ') for x in lines]

        assert transforms_fixed is not None, 'OpenRooms: Need a transforms_fixed!'
        self.transforms_fixed = transforms_fixed
        self.transforms_resize = transforms_resize
        self.transforms_matseg = transforms_matseg
        self.logger = logger
        # self.target_hw = (cfg.DATA.im_height, cfg.DATA.im_width) # if split in ['train', 'val', 'test'] else (args.test_h, args.test_w)
        self.im_width, self.im_height = self.cfg.DATA.im_width, self.cfg.DATA.im_height

        matG1File = self.opt.cfg.PATH.matcls_matIdG1_path
        matG1Dict = {}
        with open(matG1File, 'r') as f:
            for line in f.readlines():
                if 'Material__' not in line:
                    continue
                matName, mId = line.strip().split(' ')
                matG1Dict[int(mId)] = matName
        self.matG1Dict = matG1Dict
        self.opt.matG1Dict = matG1Dict

        sup_mat_lists_path = Path('/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/DatasetCreation/MatLists')
        sup_mat_lists = [x for x in sup_mat_lists_path.iterdir() if '.txt' in str(x)]
        self.sup_mat_dicts = {}
        for sup_mat_list in sorted(sup_mat_lists):
        #     print(sup_mat_list.name)
            with open(str(sup_mat_list), 'r') as fIn:
                lines = fIn.readlines()
                lines = [x.strip() for x in lines]
            self.sup_mat_dicts[sup_mat_list.stem] = lines
            
        valid_sup_classes = ['fabric', 'leather', 'metal', 'paint', 'plastic', 'rough_stone', 'rubber', 'specular_stone', 'wood']
        self.valid_sup_classes_dict = {idx+1: valid_sup_classes[idx] for idx in range(len(valid_sup_classes))}
        self.valid_sup_classes_dict.update({0: 'N/A'})
        opt.valid_sup_classes_dict = self.valid_sup_classes_dict

        self.sup_mat_dicts = {x: self.sup_mat_dicts[x] for x in self.sup_mat_dicts if x in valid_sup_classes}
        assert opt.cfg.MODEL_MATCLS.num_classes_sup == len(self.sup_mat_dicts.keys())

        self.mat_to_supcls_dict = {}
        for supcls_id, supcls_name in enumerate(self.sup_mat_dicts.keys()):
            for mat in self.sup_mat_dicts[supcls_name]:
        #         print(supcls_id, supcls, mat)
                self.mat_to_supcls_dict[mat] = [supcls_id+1, supcls_name] # supcls==0 for unlabelled
        self.mat_to_supcls_dict_keys = list(self.mat_to_supcls_dict.keys())
                

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index):

        png_image_path, mask_path = self.data_list[index]

        image = Image.open(str(png_image_path))
        im_fixedscale_SDR_uint8 = np.array(image)
        im_fixedscale_SDR_uint8 = cv2.resize(im_fixedscale_SDR_uint8, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )

        image_transformed_fixed = self.transforms_fixed(im_fixedscale_SDR_uint8)
        im_trainval_SDR = self.transforms_resize(im_fixedscale_SDR_uint8) # not necessarily \in [0., 1.] [!!!!]
        im_SDR_RGB = im_fixedscale_SDR_uint8.astype(np.float32) / 255.
        im_trainval = im_SDR_RGB

        batch_dict = {'image_path': str(png_image_path), 'image_index': index}
        batch_dict.update({'image_transformed_fixed': image_transformed_fixed, 'im_trainval': torch.from_numpy(im_trainval), 'im_trainval_SDR': im_trainval_SDR, 'im_SDR_RGB': im_SDR_RGB, 'im_fixedscale_SDR_uint8': im_fixedscale_SDR_uint8})

        # ====== matcls =====
        if self.opt.cfg.DATA.load_matcls_gt:
            matMask = np.array(Image.open(str(mask_path)))[np.newaxis, :, :]
            matMask = matMask!=0
            # print(np.amax(matMask), np.amin(matMask), matMask.shape, matMask.dtype)
            mat_cls_dict = {
                'matMask': matMask,
                'matName': 'N/A',
                'matLabel': 0,
                'matLabelSup': 0, 
                'matNameSup': 'N/A'
        }
            batch_dict.update(mat_cls_dict)
        
        return batch_dict

default_collate = torch.utils.data.dataloader.default_collate
def collate_fn_OR(batch):
    """
    Data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batches: List of loaded elements via Dataset.__getitem__
    """
    collated_batch = {}
    # iterate over keys
    # print(batch[0].keys())
    for key in batch[0]:
        if key == 'boxes_batch':
            collated_batch[key] = dict()
            for subkey in batch[0][key]:
                if subkey in ['bdb2D_full', 'bdb3D_full']: # lists of original & more information (e.g. color)
                    continue
                if subkey == 'mask':
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    # print(subkey)
                    list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                    tensor_batch = torch.cat(list_of_tensor)
                collated_batch[key][subkey] = tensor_batch
        elif key in ['boxes_valid_list', 'emitter2wall_assign_info_list', 'emitters_obj_list', 'gt_layout_RAW', 'cell_info_grid']:
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            try:
                collated_batch[key] = default_collate([elem[key] for elem in batch])
            except TypeError:
                print('[!!!!] Type error in collate_fn_OR: ', key)

    return collated_batch

def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    else:
        return elem
