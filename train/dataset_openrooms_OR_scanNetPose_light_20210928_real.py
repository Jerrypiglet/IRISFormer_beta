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

import warnings
warnings.filterwarnings("ignore")

from utils import transform

from utils_dataset_openrooms_OR_scanNetPose_light_20210928 import *
class openrooms(data.Dataset):
    def __init__(self, opt, data_list=None, logger=basic_logger(), transforms_fixed=None, transforms_semseg=None, transforms_matseg=None, transforms_resize=None, 
            split='train', task=None, if_for_training=True, load_first = -1, rseed = 1, 
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
        self.dataset_name = self.cfg.DATASET.dataset_name
        self.split = split
        # assert self.split in ['train', 'val', 'test']
        assert self.split in ['val']
        self.task = self.split if task is None else task
        self.if_for_training = if_for_training

        self.data_root = self.opt.cfg.DATASET.real_images_root_path
        data_list_path = os.path.join(self.cfg.PATH.root, self.cfg.DATASET.real_images_list_path)
        self.data_list = make_dataset_real(opt, self.data_root, data_list_path, logger=self.logger)

        logger.info(white_blue('%s: total frames: %d'%(self.dataset_name, len(self.data_list))))

        self.cascadeLevel = cascadeLevel

        assert transforms_fixed is not None, 'OpenRooms: Need a transforms_fixed!'
        self.transforms_fixed = transforms_fixed
        self.transforms_resize = transforms_resize
        self.transforms_matseg = transforms_matseg
        self.transforms_semseg = transforms_semseg

        self.logger = logger
        # self.target_hw = (cfg.DATA.im_height, cfg.DATA.im_width) # if split in ['train', 'val', 'test'] else (args.test_h, args.test_w)
        self.im_width, self.im_height = self.cfg.DATA.im_width, self.cfg.DATA.im_height
        self.im_height_padded, self.im_width_padded = self.cfg.DATA.im_height_padded_to, self.cfg.DATA.im_width_padded_to

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        png_image_path = self.data_list[index]
        frame_info = {'index': index, 'png_image_path': png_image_path}
        batch_dict = {'image_index': index}

        im_height_padded, im_width_padded = self.im_height_padded, self.im_width_padded

        hdr_scale = 1.
        # Read PNG image
        image = Image.open(str(png_image_path))
        im_fixedscale_SDR_uint8 = np.array(image)
        im_h, im_w = im_fixedscale_SDR_uint8.shape[0], im_fixedscale_SDR_uint8.shape[1]
        if not self.opt.cfg.DATA.if_pad_to_32x:
            im_height_padded, im_width_padded = int(np.ceil(float(im_h)/4.)*4), int(np.ceil(float(im_w)/4.)*4)
            print('>>>>', im_height_padded, im_width_padded)
            assert self.opt.cfg.TEST.ims_per_batch == 1

        pad_mask = np.zeros((im_height_padded, im_width_padded), dtype=np.uint8)
        if float(im_h) / float(im_w) < float(im_height_padded) / float(im_width_padded): # flatter
            im_w_resized_to = im_width_padded
            im_h_resized_to = int(float(im_h) / float(im_w) * im_w_resized_to)
            assert im_h_resized_to <= im_height_padded
            pad_mask[:im_h_resized_to, :] = 1
        else: # taller
            im_h_resized_to = im_height_padded
            im_w_resized_to = int(float(im_w) / float(im_h) * im_h_resized_to)
            assert im_w_resized_to <= im_width_padded
            pad_mask[:, :im_w_resized_to] = 1

        im_fixedscale_SDR_uint8 = cv2.resize(im_fixedscale_SDR_uint8, (im_w_resized_to, im_h_resized_to), interpolation = cv2.INTER_AREA )
        # print(im_w_resized_to, im_h_resized_to, im_w, im_h)
        assert self.opt.cfg.DATA.pad_option == 'const'
        im_fixedscale_SDR_uint8 = cv2.copyMakeBorder(im_fixedscale_SDR_uint8, 0, im_height_padded-im_h_resized_to, 0, im_width_padded-im_w_resized_to, cv2.BORDER_CONSTANT, value=0)
        # print(im_fixedscale_SDR_uint8.shape, pad_mask.shape)
        im_fixedscale_SDR = im_fixedscale_SDR_uint8.astype(np.float32) / 255.
        im_fixedscale_SDR = im_fixedscale_SDR.transpose(2, 0, 1)


        if self.opt.cfg.DATA.if_load_png_not_hdr:
            # [PNG]
            assert False, 'all models are trained with HDR input for now; should convert real images to HDR images by ** 2.2'
            im_fixedscale_HDR = (im_fixedscale_SDR - 0.5) / 0.5
            im_trainval = torch.from_numpy(im_fixedscale_HDR) # channel first for training
            im_trainval_SDR = torch.from_numpy(im_fixedscale_SDR)
            im_fixedscale_SDR = torch.from_numpy(im_fixedscale_SDR)
        else:
            # [HDR]
            im_fixedscale_HDR = im_fixedscale_SDR ** 2.2
            im_trainval = torch.from_numpy(im_fixedscale_HDR) # channel first for training
            im_trainval_SDR = torch.from_numpy(im_fixedscale_SDR)
            im_fixedscale_SDR = torch.from_numpy(im_fixedscale_SDR)

        # image_transformed_fixed = self.transforms_fixed(im_fixedscale_SDR_uint8)
        # im_trainval_SDR = self.transforms_resize(im_fixedscale_SDR_uint8) # not necessarily \in [0., 1.] [!!!!]; already padded
        # # print(im_trainval_SDR.shape, type(im_trainval_SDR), torch.max(im_trainval_SDR), torch.min(im_trainval_SDR), torch.mean(im_trainval_SDR))
        # im_trainval = torch.from_numpy(im_fixedscale_HDR) # channel first for training

        batch_dict.update({'image_path': str(png_image_path), 'pad_mask': pad_mask, 'brdf_loss_mask': pad_mask})
        batch_dict['frame_info'] = {'image_path': str(png_image_path)}
        batch_dict.update({'im_w_resized_to': im_w_resized_to, 'im_h_resized_to': im_h_resized_to})
        # batch_dict.update({'hdr_scale': hdr_scale, 'image_transformed_fixed': image_transformed_fixed, 'im_trainval': im_trainval, 'im_trainval_SDR': im_trainval_SDR, 'im_fixedscale_SDR': im_fixedscale_SDR, 'im_fixedscale_SDR_uint8': im_fixedscale_SDR_uint8})
        batch_dict.update({'hdr_scale': hdr_scale, 'im_trainval': im_trainval, 'im_trainval_SDR': im_trainval_SDR, 'im_fixedscale_SDR': im_fixedscale_SDR.permute(1, 2, 0)}) # im_fixedscale_SDR for Tensorboard logging

        if self.cfg.DEBUG.if_load_dump_BRDF_offline:
            real_sample_name = str(png_image_path).split('/')[-2]
            scene_path_dump = Path(self.opt.cfg.DEBUG.dump_BRDF_offline.path_task) / real_sample_name

            if 'al' in self.cfg.DATA.data_read_list:
                if self.opt.cfg.MODEL_BRDF.use_scale_aware_albedo:
                    albedo_path = scene_path_dump / ('imbaseColor.png')
                else:
                    albedo_path = scene_path_dump / ('imbaseColor_scaleInv.png')
                frame_info['albedo_path'] = albedo_path
                albedo = np.asarray(Image.open(albedo_path), dtype=np.float32) / 255.
                albedo = np.transpose(albedo, [2, 0, 1] )
                albedo = albedo ** 2.2
                batch_dict.update({'albedo': torch.from_numpy(albedo)})

            if 'no' in self.cfg.DATA.data_read_list:
                normal_path = scene_path_dump / ('imnormal.png')
                frame_info['normal_path'] = normal_path
                normal = np.asarray(Image.open(normal_path), dtype=np.float32) / 255.
                normal = np.transpose(normal, [2, 0, 1] )
                normal = normal * 2. - 1.
                batch_dict.update({'normal': torch.from_numpy(normal),})

            if 'ro' in self.cfg.DATA.data_read_list:
                rough_path = scene_path_dump / ('imroughness.png')
                frame_info['rough_path'] = rough_path
                # Read roughness
                rough = np.asarray(Image.open(rough_path), dtype=np.float32) / 255.
                # assert False
                rough = rough * 2. - 1.
                rough = np.expand_dims(rough, 0)
                batch_dict.update({'rough': torch.from_numpy(rough),})

            if 'de' in self.cfg.DATA.data_read_list or 'de' in self.cfg.DATA.data_read_list:
                if self.cfg.MODEL_BRDF.use_scale_aware_depth:
                    depth_path = scene_path_dump / ('imdepth.pickle')
                else:
                    depth_path = scene_path_dump / ('imdepth_scale_invariant.pickle')
                frame_info['depth_path'] = depth_path
                # Read depth
                with open(depth_path, 'rb') as f:
                    depth_dict = pickle.load(f)
                depth = depth_dict['depth_pred']
                depth = np.expand_dims(depth, 0)
                batch_dict.update({'depth': torch.from_numpy(depth),})

        return batch_dict

    def loadImage(self, imName, isGama = False):
        if not(osp.isfile(imName ) ):
            self.logger.warning('File does not exist: ' + imName )
            assert(False), 'File does not exist: ' + imName 

        im = Image.open(imName)
        im = im.resize([self.im_width, self.im_height], Image.ANTIALIAS )

        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1] )

        return im

    def loadHdr(self, imName):
        if not(osp.isfile(imName ) ):
            if osp.isfile(imName.replace('.hdr', '.rgbe')):
                imName = imName.replace('.hdr', '.rgbe')
            else:
                print(imName )
                assert(False )
        im = cv2.imread(imName, -1)
        # print(imName, im.shape, im.dtype)

        if im is None:
            print(imName )
            assert(False )
        im = cv2.resize(im, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :]
        return im

    def scaleHdr(self, hdr, seg, forced_fixed_scale=False, if_print=False):
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.split == 'train' and not forced_fixed_scale:
            # print('randommmm', np.random.random(), random.random())
            # scale = (0.95 - 0.1 * np.random.random() )  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
            scale = (0.95 - 0.1 * random.random() )  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
        else:
            scale = (0.95 - 0.05)  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
            # if if_print:
            #     print(self.split, not forced_fixed_scale, scale)

        # print('-', hdr.shape, np.max(hdr), np.min(hdr), np.median(hdr), np.mean(hdr))
        # print('----', seg.shape, np.max(seg), np.min(seg), np.median(seg), np.mean(seg))
        # print('-------', scale)
        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale 


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
                if subkey in ['mask', 'random_id', 'cat_name']: # list of lists
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                    try:
                        tensor_batch = torch.cat(list_of_tensor)
                        # print(subkey, [x['boxes_batch'][subkey].shape for x in batch], tensor_batch.shape)
                    except RuntimeError:
                        print(subkey, [x.shape for x in list_of_tensor])
                collated_batch[key][subkey] = tensor_batch
        elif key in ['frame_info', 'image_index']:
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            try:
                collated_batch[key] = default_collate([elem[key] for elem in batch])
            except RuntimeError as e:
                print('[!!!!] Type error in collate_fn_OR: ', key, e)
                # print(type(batch[0][key]))
                # print(batch[0][key].dtype)

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
