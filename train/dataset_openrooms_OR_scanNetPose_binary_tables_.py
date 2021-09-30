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
import h5py
import tables

# import math
# from detectron2.structures import BoxMode
# from detectron2.data.dataset_mapper import DatasetMapper

from utils.utils_total3D.utils_OR_vis_labels import RGB_to_01
from utils.utils_total3D.utils_others import Relation_Config, OR4XCLASSES_dict, OR4XCLASSES_not_detect_mapping_ids_dict, OR4X_mapping_catInt_to_RGB
# from detectron2.data import build_detection_test_loader,DatasetCatalog, MetadataCatalog

from utils.utils_scannet import read_ExtM_from_txt, read_img
import utils.utils_nvidia.mdataloader.m_preprocess as m_preprocess
import PIL
import torchvision.transforms as tfv_transform

from itertools import cycle

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

def make_dataset(opt, split, task, data_root=None, data_list=None, logger=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Scene list file do not exist: " + data_list + "\n"))
    if logger is None:
        logger = basic_logger()
        
    meta_split_scene_name_list = []
    list_read = open(data_list).readlines()
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

    if opt.cfg.DATASET.first_scenes != -1:
        return meta_split_scene_name_list[:opt.cfg.DATASET.first_scenes]
    elif opt.cfg.DATASET.if_quarter and task != 'vis':
        return return_percent(meta_split_scene_name_list, 0.25)
    else:
        return meta_split_scene_name_list

def get_per_frame_dataset_info(opt, split='train', data_root=None, data_list=None, valid_scene_key_list=None, logger=None):
    assert split in ['train', 'val', 'test']
    data_list = data_list.replace('_scenes.txt', '_scenes_frame_info.txt')
    if not os.path.isfile(data_list):
        raise (RuntimeError("Sample list file do not exist: " + data_list + "\n"))
    if logger is None:
        logger = basic_logger()
    
    frame_info_list = []
    list_read = open(data_list).readlines()
    # logger.info("Totally {} samples in {} set.".format(len(list_read), split))
    for line in list_read:
        line = line.strip()
        scene_key, frame_id = line.split(' ')[0], int(line.split(' ')[1])
    #     if split == 'test':
    #         assert False, 'No support for test split for now.'
    #     else:
    #         if len(line_split) not in [2]:
    #             raise (RuntimeError("Scene list file read line error : " + line + "\n"))
        meta_split, scene_name = scene_key.split('-')
        if valid_scene_key_list is not None and scene_key not in valid_scene_key_list:
            continue

        frame_info_list.append([scene_key, frame_id])

    return frame_info_list

class openrooms_binary(data.IterableDataset):
    def __init__(self, opt, logger=basic_logger(), transforms_fixed=None, transforms_semseg=None, transforms_matseg=None, transforms_resize=None, 
            split='train', task=None, if_for_training=True, rseed = 1, load_first = -1, 
            cascadeLevel = 0):

        if logger is None:
            logger = basic_logger()

        self.opt = opt
        self.cfg = self.opt.cfg
        self.logger = logger
        self.rseed = rseed
        self.dataset_name = self.cfg.DATASET.dataset_name
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.task = self.split if task is None else task
        self.if_for_training = if_for_training
        self.data_root = self.opt.cfg.DATASET.dataset_path_binary
        split_to_list = {'train': 'train_scenes.txt', 'val': 'val_scenes.txt', 'test': 'test_scenes.txt'}
        data_list_file = os.path.join(self.cfg.PATH.root, self.cfg.DATASET.dataset_list)
        data_list_file = os.path.join(data_list_file, split_to_list[split])
        self.meta_split_scene_name_list = make_dataset(opt, split, self.task, self.data_root, data_list_file, logger=self.logger)

        self.scene_key_frame_id_list = get_per_frame_dataset_info(opt, split, self.data_root, data_list_file, valid_scene_key_list=['-'.join(_) for _ in self.meta_split_scene_name_list], logger=self.logger)
        self.num_frames = len(self.scene_key_frame_id_list)
        # self.start = 0
        # self.end = self.num_frames - 1
        self.world_size = self.opt.num_gpus
        self.rank = self.opt.rank

        rank_split_num_frames = len(self.scene_key_frame_id_list) // self.world_size
        self.scene_key_frame_id_list_this_rank = self.scene_key_frame_id_list[self.rank*rank_split_num_frames : (self.rank+1)*rank_split_num_frames]

        logger.info(white_blue('%s-%s: total scenes: %d; %d samples'%(self.dataset_name, self.split, len(self.meta_split_scene_name_list), self.num_frames)))

        self.OR = opt.cfg.MODEL_LAYOUT_EMITTER.data.OR
        self.valid_class_ids = RECON_3D_CLS_OR_dict[self.OR]

        self.cascadeLevel = cascadeLevel

        assert transforms_fixed is not None, 'OpenRooms: Need a transforms_fixed!'
        self.transforms_fixed = transforms_fixed
        self.transforms_resize = transforms_resize
        self.transforms_matseg = transforms_matseg
        self.transforms_semseg = transforms_semseg

        self.logger = logger
        self.im_height, self.im_width = self.cfg.DATA.im_height, self.cfg.DATA.im_width
        self.if_resize = (self.opt.cfg.DATA.im_height_ori, self.opt.cfg.DATA.im_width_ori) == (self.opt.cfg.DATA.im_height_ori, self.opt.cfg.DATA.im_width_ori)

        self.OR = self.cfg.MODEL_LAYOUT_EMITTER.data.OR
        self.OR_classes = OR4XCLASSES_dict[self.OR]

        if self.opt.cfg.MODEL_GMM.enable:
            self.to_gray = tfv_transform.Compose( [tfv_transform.Grayscale(),
                tfv_transform.ToTensor()])
            self.T_to_tensor = tfv_transform.ToTensor()


    def __len__(self):
        # return len(self.meta_split_scene_name_list)
        return self.num_frames

    # def __iter__(self):
    #     return iter(cycle(self.yield_sample))

    # def yield_sample(self):
    # def __iter__(self):
    #     # idx = 0
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None or worker_info.num_workers==1:
    #         meta_split_scene_name_list_workers = [self.meta_split_scene_name_list]
    #         worker_id = 0
    #     else:
    #         meta_split_scene_name_list_workers = [list(_) for _ in np.array_split(self.meta_split_scene_name_list, worker_info.num_workers)]
    #         worker_id = worker_info.id

    #     meta_split_scene_name_list_per_worker = meta_split_scene_name_list_workers[worker_id]
    #     for meta_split, scene_name in meta_split_scene_name_list_per_worker:
    #         im_png_h5 = Path(self.opt.cfg.DATASET.dataset_path) / meta_split / scene_name / 'im_png.h5'
    #         assert im_png_h5.exists(), '%s does not exist!'%(str(im_png_h5))
    #         with h5py.File(str(im_png_h5), 'r') as hf:
    #             sample_id_list = np.array(hf.get('sample_id_list'))
    #             im_uint8_array = np.array(hf.get('im_uint8'))
    #             seg_uint8_array = np.array(hf.get('seg_uint8'))

    #         for frame_id in sample_id_list:
    #             frame_info = {'meta_split': meta_split, 'scene_name': scene_name, 'frame_id': frame_id}
    #             batch_dict = {'frame_info': frame_info}
    #             # idx += 1
    #             # print('======', worker_info.id, len(meta_split_scene_name_list_per_worker))
    #             yield batch_dict

    def __iter__(self): # https://gist.github.com/kklemon/c745e9ee2474f6907f2a3189c0da68b5  
        index = 0
        worker_info = torch.utils.data.get_worker_info()
        # mod = self.world_size
        # shift = self.rank
        # if worker_info:
        #     mod *= worker_info.num_workers
        #     shift = self.rank * worker_info.num_workers + worker_info.id


        scenes_this_rank = self.meta_split_scene_name_list

        if worker_info is None or worker_info.num_workers==1:
            meta_split_scene_name_list_workers = [scenes_this_rank]
            worker_id = 0
        else:
            meta_split_scene_name_list_workers = [list(_) for _ in np.array_split(scenes_this_rank, worker_info.num_workers)]
            worker_id = worker_info.id
        meta_split_scene_name_list_per_worker = meta_split_scene_name_list_workers[worker_id]

        # meta_split_scene_name_list_per_worker = [self.meta_split_scene_name_list[scene_idx] for scene_idx in range(len(self.meta_split_scene_name_list)) if (scene_idx + shift) % mod == 0]

        # print('>>>>', self.rank, worker_info.id, len(scenes_this_rank), '<<<', len(meta_split_scene_name_list_per_worker))

        # meta_split_scene_name_list_per_worker = self.meta_split_scene_name_list

        if_load_immask = self.opt.cfg.DATA.load_brdf_gt and (not self.opt.cfg.DATASET.if_no_gt_semantics)

        for meta_split, scene_name in meta_split_scene_name_list_per_worker:
            im_png_h5 = Path(self.data_root) / 'im_png' / meta_split / scene_name / 'im_png.h5'
            assert im_png_h5.exists(), '%s does not exist!'%(str(im_png_h5))
            try:
                # with h5py.File(str(im_png_h5), 'r') as hf:
                #     sample_id_list = np.array(hf.get('sample_id_list'))
                #     im_uint8_array = np.array(hf.get('im_uint8'))
                #     if if_load_immask:
                #         seg_uint8_array = np.array(hf.get('seg_uint8'))
                #         mask_int32_array = np.array(hf.get('mask_int32'))
                #     if self.opt.cfg.DATASET.binary.if_in_one_file:
                #         if 'al' in self.cfg.DATA.data_read_list:
                #             albedo_uint8_array = np.array(hf.get('albedo_uint8'))
                #         if 'de' in self.cfg.DATA.data_read_list:
                #             depth_float32_array = np.array(hf.get('depth_float32'))
                print(str(im_png_h5))
                h5file = tables.open_file(str(im_png_h5), driver="H5FD_CORE")
                sample_id_list = h5file.root.sample_id_list.read()
                im_uint8_array = h5file.root.im_uint8.read()
                if if_load_immask:
                    seg_uint8_array = h5file.root.seg_uint8.read()
                    mask_int32_array = h5file.root.mask_int32.read().astype(np.int32)
                if self.opt.cfg.DATASET.binary.if_in_one_file:
                    try:
                        if 'al' in self.cfg.DATA.data_read_list:
                            albedo_uint8_array = h5file.root.albedo_uint8.read()
                        if 'de' in self.cfg.DATA.data_read_list:
                            depth_float32_array = h5file.root.depth_float32.read()
                    except:
                        print(str(im_png_h5))

                h5file.close()                


            except OSError:
                print('[!!!!!!] Error reading '+str(im_png_h5))

            brdf_batch_dict = {}
            if 'al' in self.cfg.DATA.data_read_list:
                if not self.opt.cfg.DATASET.binary.if_in_one_file:
                    albedo_h5 = Path(self.data_root) / 'albedo' / meta_split / scene_name / 'albedo.h5'
                    assert albedo_h5.exists(), '%s does not exist!'%(str(albedo_h5))
                    # with h5py.File(str(albedo_h5), 'r') as hf:
                    #     albedo_uint8_array = np.array(hf.get('albedo_uint8'))
                    h5file = tables.open_file(str(albedo_h5), driver="H5FD_CORE")
                    albedo_uint8_array = h5file.root.albedo_uint8.read()
                    h5file.close()
                brdf_batch_dict['albedo'] = albedo_uint8_array

            if 'de' in self.cfg.DATA.data_read_list:
                if not self.opt.cfg.DATASET.binary.if_in_one_file:
                    depth_h5 = Path(self.data_root) / 'depth' / meta_split / scene_name / 'depth.h5'
                    assert depth_h5.exists(), '%s does not exist!'%(str(depth_h5))
                    # with h5py.File(str(depth_h5), 'r') as hf:
                    #     depth_float32_array = np.array(hf.get('depth_float32'))
                    h5file = tables.open_file(str(depth_h5), driver="H5FD_CORE")
                    depth_float32_array = h5file.root.depth_float32.read()
                    h5file.close()

                brdf_batch_dict['depth'] = depth_float32_array

            for in_batch_idx, frame_id in enumerate(sample_id_list):
                scene_key = '-'.join([meta_split, scene_name])
                frame_info = {'scene_key': scene_key, 'meta_split': meta_split, 'scene_name': scene_name, 'frame_id': frame_id}
                batch_dict = {'frame_info': frame_info}
                if [scene_key, frame_id] not in self.scene_key_frame_id_list_this_rank:
                    continue
                
                scene_total3d_path = Path(self.cfg.DATASET.layout_emitter_path) / meta_split / scene_name
                frame_info = {'index': index, 'meta_split': meta_split, 'scene_name': scene_name, 'frame_id': frame_id, 'frame_key': '%s-%s-%d'%(meta_split, scene_name, frame_id), \
                    'scene_total3d_path': scene_total3d_path}
                batch_dict = {'image_index': index, 'frame_info': frame_info, 'image_path': ''}

                if_load_immask = if_load_immask or self.opt.cfg.MODEL_MATSEG.enable
                # if_load_immask = False
                self.opt.if_load_immask = if_load_immask

                if self.opt.cfg.DATA.if_pad_to_32x:
                    assert if_load_immask

                mask_path = ''
                if if_load_immask:
                    # seg_path = hdr_image_path.replace('im_', 'immask_').replace('hdr', 'png').replace('DiffMat', '')
                    # # Read segmentation
                    # seg = 0.5 * (self.loadImage(seg_path ) + 1)[0:1, :, :]
                    seg = 0.5 * (self.loadImage(im=seg_uint8_array[in_batch_idx] ) + 1)[0:1, :, :]
                    # semantics_path = hdr_image_path.replace('DiffMat', '').replace('DiffLight', '')
                    # mask_path = semantics_path.replace('im_', 'imcadmatobj_').replace('hdr', 'dat')
                    # print(mask_int32_array.shape, sample_id_list, meta_split, scene_name)
                    # print(mask_int32_array[in_batch_idx].shape)
                    mask = self.loadBinary(im=mask_int32_array[in_batch_idx]).squeeze() # [h, w, 3]
                else:
                    seg = np.ones((1, self.im_height, self.im_width), dtype=np.float32)
                    mask = np.ones((self.im_height, self.im_width, 3), dtype=np.uint8)

                brdf_loss_mask = np.ones((self.im_height, self.im_width), dtype=np.uint8)
                if self.opt.if_pad:
                    mask = self.opt.pad_op(mask, name='mask')
                    seg = self.opt.pad_op(seg, if_channel_first=True, name='seg')
                    brdf_loss_mask = self.opt.pad_op(brdf_loss_mask, if_channel_2_input=True, name='brdf_loss_mask')

                hdr_scale = 1.

                # Read PNG image
                # image = Image.open(str(png_image_path))
                # im_fixedscale_SDR_uint8 = np.array(image)
                im_fixedscale_SDR_uint8 = im_uint8_array[in_batch_idx]
                # im_fixedscale_SDR_uint8 = cv2.resize(im_fixedscale_SDR_uint8, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )

                image_transformed_fixed = self.transforms_fixed(im_fixedscale_SDR_uint8)
                im_trainval_SDR = self.transforms_resize(im_fixedscale_SDR_uint8) # not necessarily \in [0., 1.] [!!!!]
                # print(im_trainval_SDR.shape, type(im_trainval_SDR), torch.max(im_trainval_SDR), torch.min(im_trainval_SDR), torch.mean(im_trainval_SDR))
                im_fixedscale_SDR = im_fixedscale_SDR_uint8.astype(np.float32) / 255.
                if self.opt.if_pad:
                    im_fixedscale_SDR = self.opt.pad_op(im_fixedscale_SDR, name='im_fixedscale_SDR')

                im_trainval = im_trainval_SDR # [3, 240, 320], tensor, not in [0., 1.]

                batch_dict.update({'brdf_loss_mask': torch.from_numpy(brdf_loss_mask)})

                if self.opt.cfg.DATA.if_also_load_next_frame:
                    assert False, 'currently not supported'
                    png_image_next_path = Path(self.opt.cfg.DATASET.png_path) / meta_split / scene_name / ('im_%d.png'%(frame_id+1))
                    if not png_image_next_path.exists():
                        return self.__getitem__((index+1)%len(self.meta_split_scene_name_list))
                    image_next = Image.open(str(png_image_next_path))
                    im_fixedscale_SDR_uint8_next = np.array(image_next)
                    im_fixedscale_SDR_uint8_next = cv2.resize(im_fixedscale_SDR_uint8_next, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )
                    im_fixedscale_SDR_next = im_fixedscale_SDR_uint8_next.astype(np.float32) / 255.
                    batch_dict.update({'im_fixedscale_SDR_next': im_fixedscale_SDR_next})

                
                # image_transformed_fixed: normalized, not augmented [only needed in semseg]

                # im_trainval: normalized, augmented; HDR (same as im_trainval_SDR in png case) -> for input to network

                # im_trainval_SDR: normalized, augmented; LDR (SRGB space)
                # im_fixedscale_SDR: normalized, NOT augmented; LDR
                # im_fixedscale_SDR_uint8: im_fixedscale_SDR -> 255

                # print('------', image_transformed_fixed.shape, im_trainval.shape, im_trainval_SDR.shape, im_fixedscale_SDR.shape, im_fixedscale_SDR_uint8.shape, )
                # png: ------ torch.Size([3, 240, 320]) (240, 320, 3) torch.Size([3, 240, 320]) (240, 320, 3) (240, 320, 3)
                # hdr: ------ torch.Size([3, 240, 320]) (3, 240, 320) (3, 240, 320) (3, 240, 320) (240, 320, 3)

                batch_dict.update({'hdr_scale': hdr_scale, 'image_transformed_fixed': image_transformed_fixed, 'im_trainval': im_trainval, 'im_trainval_SDR': im_trainval_SDR, 'im_fixedscale_SDR': im_fixedscale_SDR, 'im_fixedscale_SDR_uint8': im_fixedscale_SDR_uint8})

                # ====== BRDF =====
                # image_path = batch_dict['image_path']
                # if self.opt.cfg.DATA.load_brdf_gt and (not self.opt.cfg.DATASET.if_no_gt_semantics):
                if self.opt.cfg.DATA.load_brdf_gt:
                    batch_dict_brdf = self.load_brdf_lighting('', if_load_immask, '', mask, seg, hdr_scale, frame_info, brdf_batch_dict=brdf_batch_dict, in_batch_idx=in_batch_idx)
                    batch_dict.update(batch_dict_brdf)

                if self.opt.cfg.MODEL_GMM.enable:
                    self.load_scannet_compatible(batch_dict, frame_info)

                # ====== matseg =====
                if self.opt.cfg.DATA.load_matseg_gt:
                    mat_seg_dict = self.load_matseg(mask, im_fixedscale_SDR_uint8)
                    batch_dict.update(mat_seg_dict)

                index += 1
                yield batch_dict


    def load_scannet_compatible(self, batch_dict, frame_info):
        meta_split, scene_name, frame_id = frame_info['meta_split'], frame_info['scene_name'], frame_info['frame_id']
        if self.opt.cfg.DATA.load_cam_pose:
            # if loading OR cam.txt files: need to inverse of def computeCameraEx() /home/ruizhu/Documents/Projects/Total3DUnderstanding/utils_OR/DatasetCreation/sampleCameraPoseFromScanNet.py
            cam_txt_path = Path(self.data_root) / meta_split / scene_name / ('pose_%d.txt'%frame_id)
            pose_ExtM = read_ExtM_from_txt(cam_txt_path)
            batch_dict.update({'pose_ExtM': pose_ExtM})

        img_path = frame_info['png_image_path']
        proc_normalize = m_preprocess.get_transform()
        proc_totensor = m_preprocess.to_tensor()

        img = read_img(img_path , no_process = True)[0]
        img_raw = self.T_to_tensor( img )
        # print('--', img.size, img_raw.shape, img_path) # -- (320, 240) torch.Size([3, 240, 320]) /data/ruizhu/OR-pngs/main_xml1/scene0610_01/im_12.png

        img = img.resize([self.im_width, self.im_height], PIL.Image.NEAREST) 
        img_small = img.resize( [self.im_width//4, self.im_height//4], PIL.Image.NEAREST )
        img_gray = self.to_gray(img) 

        img = proc_normalize(img)
        img_small = proc_normalize( img_small )
        # print('-->', img.shape, img_raw.shape, img_small.shape) # --> torch.Size([3, 240, 320]) torch.Size([3, 240, 320]) torch.Size([3, 60, 80])

        batch_dict.update({'img_GMM': img.unsqueeze_(0), 
            'img_raw_GMM': img_raw.unsqueeze_(0),
            'img_small_GMM': img_small.unsqueeze_(0)})

        # scene_dict = self.read_scene(frame_info=frame_info)
        # cam_K = scene_dict['camera']['K'] # [[577.8708   0.     320.    ], [  0.     577.8708 240.    ], [  0.       0.       1.    ]]
        # cam_K_ratio_W = cam_K[0][2] / (self.im_width/2.)
        # cam_K_ratio_H = cam_K[1][2] / (self.im_height/2.)
        # assert cam_K_ratio_W == cam_K_ratio_H
        # cam_K_scaled = np.vstack([cam_K[:2, :] / cam_K_ratio_W, cam_K[2:3, :]])
        # batch_dict.update({'cam_K_scaled_GMM': cam_K_scaled})
        # print(cam_K_scaled)


    def load_brdf_lighting(self, hdr_image_path, if_load_immask, mask_path, mask, seg, hdr_scale, frame_info, brdf_batch_dict=None, in_batch_idx=-1):
        batch_dict_brdf = {}
        # Get paths for BRDF params
        if 'al' in self.cfg.DATA.data_read_list:
            # albedo_path = hdr_image_path.replace('im_', 'imbaseColor_').replace('rgbe', 'png').replace('hdr', 'png')
            # if self.opt.cfg.DATASET.dataset_if_save_space:
            #     albedo_path = albedo_path.replace('DiffLight', '')
            # Read albedo
            albedo = self.loadImage(im=brdf_batch_dict['albedo'][in_batch_idx], isGama = False)
            albedo = (0.5 * (albedo + 1) ) ** 2.2
            if self.opt.if_pad:
                albedo = self.opt.pad_op(albedo, if_channel_first=True, name='albedo')

            batch_dict_brdf.update({'albedo': torch.from_numpy(albedo)})

        if 'no' in self.cfg.DATA.data_read_list:
            normal_path = hdr_image_path.replace('im_', 'imnormal_').replace('rgbe', 'png').replace('hdr', 'png')
            if self.opt.cfg.DATASET.dataset_if_save_space:
                normal_path = normal_path.replace('DiffLight', '').replace('DiffMat', '')
            # normalize the normal vector so that it will be unit length
            normal = self.loadImage(normal_path )
            normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]
            if self.opt.if_pad:
                normal = self.opt.pad_op(normal, if_channel_first=True, name='normal')

            batch_dict_brdf.update({'normal': torch.from_numpy(normal),})

        if 'ro' in self.cfg.DATA.data_read_list:
            rough_path = hdr_image_path.replace('im_', 'imroughness_').replace('rgbe', 'png').replace('hdr', 'png')
            if self.opt.cfg.DATASET.dataset_if_save_space:
                rough_path = rough_path.replace('DiffLight', '')
            # Read roughness
            rough = self.loadImage(rough_path )[0:1, :, :]
            if self.opt.if_pad:
                rough = self.opt.pad_op(rough, if_channel_first=True, name='rough')

            batch_dict_brdf.update({'rough': torch.from_numpy(rough),})

        if 'de' in self.cfg.DATA.data_read_list or 'de' in self.cfg.DATA.data_read_list:
            # depth_path = hdr_image_path.replace('im_', 'imdepth_').replace('rgbe', 'dat').replace('hdr', 'dat')
            # if self.opt.cfg.DATASET.dataset_if_save_space:
            #     depth_path = depth_path.replace('DiffLight', '').replace('DiffMat', '')
            # Read depth
            depth = self.loadBinary(im=brdf_batch_dict['depth'][in_batch_idx])
            if self.opt.if_pad:
                depth = self.opt.pad_op(depth, if_channel_first=True, name='depth')

            batch_dict_brdf.update({'depth': torch.from_numpy(depth),})
            if self.opt.cfg.DATA.if_also_load_next_frame:
                assert False
                frame_id = frame_info['frame_id']
                depth_path_next = depth_path.replace('%d.dat'%frame_id, '%d.dat'%(frame_id+1))
                depth_next = self.loadBinary(depth_path_next)
                if self.opt.if_pad:
                    depth_next = self.opt.pad_op(depth_next, if_channel_first=True, name='depth_next')

                batch_dict_brdf.update({'depth_next': torch.from_numpy(depth_next),})


        if self.cascadeLevel == 0:
            env_path = hdr_image_path.replace('im_', 'imenv_')
        else:
            env_path = hdr_image_path.replace('im_', 'imenv_')
            envPre_path = hdr_image_path.replace('im_', 'imenv_').replace('.hdr', '_%d.h5'  % (self.cascadeLevel -1) )
            
            albedoPre_path = hdr_image_path.replace('im_', 'imbaseColor_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )
            normalPre_path = hdr_image_path.replace('im_', 'imnormal_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )
            roughPre_path = hdr_image_path.replace('im_', 'imroughness_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )
            depthPre_path = hdr_image_path.replace('im_', 'imdepth_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )

            diffusePre_path = hdr_image_path.replace('im_', 'imdiffuse_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )
            specularPre_path = hdr_image_path.replace('im_', 'imspecular_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )

        if if_load_immask:
            segArea = np.logical_and(seg > 0.49, seg < 0.51 ).astype(np.float32 )
            segEnv = (seg < 0.1).astype(np.float32 )
            segObj = (seg > 0.9) 

            if self.opt.cfg.MODEL_LIGHT.enable:
                segObj = segObj.squeeze()
                segObj = ndimage.binary_erosion(segObj, structure=np.ones((7, 7) ),
                        border_value=1)
                segObj = segObj[np.newaxis, :, :]

            segObj = segObj.astype(np.float32 )
        else:
            segObj = np.ones_like(seg, dtype=np.float32)
            segEnv = np.zeros_like(seg, dtype=np.float32)
            segArea = np.zeros_like(seg, dtype=np.float32)

        if self.opt.cfg.DATA.load_light_gt:
            envmaps, envmapsInd = self.loadEnvmap(env_path )
            envmaps = envmaps * hdr_scale 
            # print(self.split, hdr_scale, np.amax(envmaps),np.amin(envmaps), np.median(envmaps))
            if self.cascadeLevel > 0: 
                envmapsPre = self.loadH5(envPre_path ) 
                if envmapsPre is None:
                    print("Wrong envmap pred")
                    envmapsInd = envmapsInd * 0 
                    envmapsPre = np.zeros((84, 120, 160), dtype=np.float32 ) 

        if self.cascadeLevel > 0:
            # Read albedo
            albedoPre = self.loadH5(albedoPre_path )
            albedoPre = albedoPre / np.maximum(np.mean(albedoPre ), 1e-10) / 3

            # normalize the normal vector so that it will be unit length
            normalPre = self.loadH5(normalPre_path )
            normalPre = normalPre / np.sqrt(np.maximum(np.sum(normalPre * normalPre, axis=0), 1e-5) )[np.newaxis, :]
            normalPre = 0.5 * (normalPre + 1)

            # Read roughness
            roughPre = self.loadH5(roughPre_path )[0:1, :, :]
            roughPre = 0.5 * (roughPre + 1)

            # Read depth
            depthPre = self.loadH5(depthPre_path )
            depthPre = depthPre / np.maximum(np.mean(depthPre), 1e-10) / 3

            diffusePre = self.loadH5(diffusePre_path )
            diffusePre = diffusePre / max(diffusePre.max(), 1e-10)

            specularPre = self.loadH5(specularPre_path )
            specularPre = specularPre / max(specularPre.max(), 1e-10)

        # if if_load_immask:
        batch_dict_brdf.update({
                'mask': torch.from_numpy(mask), 
                'maskPath': mask_path, 
                'segArea': torch.from_numpy(segArea),
                'segEnv': torch.from_numpy(segEnv),
                'segObj': torch.from_numpy(segObj),
                'object_type_seg': torch.from_numpy(seg), 
                })
        # if self.transform is not None and not self.opt.if_hdr:

        if self.opt.cfg.DATA.load_light_gt:
            batch_dict_brdf['envmaps'] = envmaps
            batch_dict_brdf['envmapsInd'] = envmapsInd
            # print(envmaps.shape, envmapsInd.shape)

            if self.cascadeLevel > 0:
                batch_dict_brdf['envmapsPre'] = envmapsPre

        if self.cascadeLevel > 0:
            batch_dict_brdf['albedoPre'] = albedoPre
            batch_dict_brdf['normalPre'] = normalPre
            batch_dict_brdf['roughPre'] = roughPre
            batch_dict_brdf['depthPre'] = depthPre

            batch_dict_brdf['diffusePre'] = diffusePre
            batch_dict_brdf['specularPre'] = specularPre

        return batch_dict_brdf

    def read_scene(self, frame_info):
        scene_total3d_path, frame_id = frame_info['scene_total3d_path'], frame_info['frame_id']
        pickle_path = str(scene_total3d_path / ('layout_obj_%d.pkl'%frame_id))
        pickle_path_reindexed = pickle_path.replace('.pkl', '_reindexed.pkl')
        with open(pickle_path, 'rb') as f:
            sequence = pickle.load(f)
        with open(pickle_path_reindexed, 'rb') as f:
            sequence_reindexed = pickle.load(f)

        camera = sequence['camera']
        
        return_scene_dict = {'sequence': sequence, 'sequence_reindexed': sequence_reindexed, 'camera': camera, 'scene_pickle_path': pickle_path}

        # read transformation matrices from RAW OR -> Total3D
        transform_to_total3d_coords_dict_path = str(scene_total3d_path / ('transform_to_total3d_coords_dict_%d.pkl'%frame_id))
        with open(transform_to_total3d_coords_dict_path, 'rb') as f:
            transform_to_total3d_coords_dict = pickle.load(f)
        transform_R_RAW2Total3D, transform_t_RAW2Total3D = transform_to_total3d_coords_dict['transform_R'], transform_to_total3d_coords_dict['transform_t']
        return_scene_dict.update({'transform_R_RAW2Total3D': transform_R_RAW2Total3D.astype(np.float32), 'transform_t_RAW2Total3D': transform_t_RAW2Total3D.astype(np.float32)})

        return return_scene_dict

    def load_matseg(self, mask, im_fixedscale_SDR_uint8):
        # >>>> Rui: Read obj mask
        mat_aggre_map, num_mat_masks = self.get_map_aggre_map(mask) # 0 for invalid region
        # if self.opt.if_pad:
        #     mat_aggre_map = self.opt.pad_op(mat_aggre_map, name='mat_aggre_map', if_channel_2_input=True)
        # if self.opt.if_pad:
        #     im_fixedscale_SDR_uint8 = self.opt.pad_op(im_fixedscale_SDR_uint8, name='im_fixedscale_SDR_uint8')
        # print(mat_aggre_map.shape, im_fixedscale_SDR_uint8.shape)
        im_matseg_transformed_trainval, mat_aggre_map_transformed = self.transforms_matseg(im_fixedscale_SDR_uint8, mat_aggre_map.squeeze()) # augmented
        # print(im_matseg_transformed_trainval.shape, mat_aggre_map_transformed.shape)
        mat_aggre_map = mat_aggre_map_transformed.numpy()[..., np.newaxis]

        h, w, _ = mat_aggre_map.shape
        gt_segmentation = mat_aggre_map
        segmentation = np.zeros([50, h, w], dtype=np.uint8)
        for i in range(num_mat_masks+1):
            if i == 0:
                # deal with backgroud
                seg = gt_segmentation == 0
                segmentation[num_mat_masks, :, :] = seg.reshape(h, w) # segmentation[num_mat_masks] for invalid mask
            else:
                seg = gt_segmentation == i
                segmentation[i-1, :, :] = seg.reshape(h, w) # segmentation[0..num_mat_masks-1] for plane instances
        return {
            'mat_aggre_map': torch.from_numpy(mat_aggre_map),  # 0 for invalid region
            # 'mat_aggre_map_reindex': torch.from_numpy(mat_aggre_map_reindex), # gt_seg
            'num_mat_masks': num_mat_masks,  
            'mat_notlight_mask': torch.from_numpy(mat_aggre_map!=0).float(),
            'instance': torch.ByteTensor(segmentation), # torch.Size([50, 240, 320])
            'semantic': 1 - torch.FloatTensor(segmentation[num_mat_masks, :, :]).unsqueeze(0), # torch.Size([50, 240, 320]) torch.Size([1, 240, 320])
            'im_matseg_transformed_trainval': im_matseg_transformed_trainval
        }
    
    def get_map_aggre_map(self, objMask):
        cad_map = objMask[:, :, 0]
        mat_idx_map = objMask[:, :, 1]        
        obj_idx_map = objMask[:, :, 2] # 3rd channel: object INDEX map

        mat_aggre_map = np.zeros_like(cad_map)
        cad_ids = np.unique(cad_map)
        num_mats = 1
        for cad_id in cad_ids:
            cad_mask = cad_map == cad_id
            mat_index_map_cad = mat_idx_map[cad_mask]
            mat_idxes = np.unique(mat_index_map_cad)

            obj_idx_map_cad = obj_idx_map[cad_mask]
            if_light = list(np.unique(obj_idx_map_cad))==[0]
            if if_light:
                mat_aggre_map[cad_mask] = 0
                continue

            # mat_aggre_map[cad_mask] = mat_idx_map[cad_mask] + num_mats
            # num_mats = num_mats + max(mat_idxs)
            cad_single_map = np.zeros_like(cad_map)
            cad_single_map[cad_mask] = mat_idx_map[cad_mask]
            for i, mat_idx in enumerate(mat_idxes):
        #         mat_single_map = np.zeros_like(cad_map)
                mat_aggre_map[cad_single_map==mat_idx] = num_mats
                num_mats += 1

        return mat_aggre_map, num_mats-1

    def loadImage(self, imName=None, im=None, isGama = False):

        if im is None:
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

        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale 

    def loadBinary(self, imName=None, im=None, channels = 1, dtype=np.float32, if_resize=True):
        if im is None:
            assert dtype in [np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
            if not(osp.isfile(imName ) ):
                assert(False ), '%s doesnt exist!'%imName
            with open(imName, 'rb') as fIn:
                hBuffer = fIn.read(4)
                height = struct.unpack('i', hBuffer)[0]
                wBuffer = fIn.read(4)
                width = struct.unpack('i', wBuffer)[0]
                dBuffer = fIn.read(4 * channels * width * height )
                if dtype == np.float32:
                    decode_char = 'f'
                elif dtype == np.int32:
                    decode_char = 'i'
                im = np.asarray(struct.unpack(decode_char * channels * height * width, dBuffer), dtype=dtype)
                im = im.reshape([height, width, channels] )
                if if_resize:
                    # print(self.im_width, self.im_height, width, height)
                    if dtype == np.float32:
                        im = cv2.resize(im, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA )
                    elif dtype == np.int32:
                        im = cv2.resize(im.astype(np.float32), (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
                        im = im.astype(np.int32)

                im = np.squeeze(im)

        return im[np.newaxis, :, :]

    def loadH5(self, imName ): 
        try:
            hf = h5py.File(imName, 'r')
            im = np.array(hf.get('data' ) )
            return im 
        except:
            return None

    def loadEnvmap(self, envName ):
        # print('>>>>loadEnvmap', envName)
        if not osp.isfile(envName ):
            env = np.zeros( [3, self.envRow, self.envCol,
                self.envHeight, self.envWidth], dtype = np.float32 )
            envInd = np.zeros([1, 1, 1], dtype=np.float32 )
            print('Warning: the envmap %s does not exist.' % envName )
            return env, envInd
        else:
            envHeightOrig, envWidthOrig = 16, 32
            assert( (envHeightOrig / self.envHeight) == (envWidthOrig / self.envWidth) )
            assert( envHeightOrig % self.envHeight == 0)
            
            env = cv2.imread(envName, -1 ) 

            if not env is None:
                env = env.reshape(self.envRow, envHeightOrig, self.envCol,
                    envWidthOrig, 3) # (1920, 5120, 3) -> (120, 16, 160, 32, 3)
                env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3] ) ) # -> (3, 120, 160, 16, 32)

                scale = envHeightOrig / self.envHeight
                if scale > 1:
                    env = block_reduce(env, block_size = (1, 1, 1, 2, 2), func = np.mean )

                envInd = np.ones([1, 1, 1], dtype=np.float32 )
                return env, envInd
            else:
                env = np.zeros( [3, self.envRow, self.envCol,
                    self.envHeight, self.envWidth], dtype = np.float32 )
                envInd = np.zeros([1, 1, 1], dtype=np.float32 )
                print('Warning: the envmap %s does not exist.' % envName )
                return env, envInd

    def loadNPY(self, imName, dtype=np.int32, if_resize=True):
        depth = np.load(imName)
        if if_resize:
            #t0 = timeit.default_timer()
            if dtype == np.float32:
                depth = cv2.resize(
                    depth, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA)
                #print('Resize float npy: %.4f' % (timeit.default_timer() - t0) )
            elif dtype == np.int32:
                depth = cv2.resize(depth.astype(
                    np.float32), (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.int32)
                #print('Resize int32 npy: %.4f' % (timeit.default_timer() - t0) )

        depth = np.squeeze(depth)

        return depth
    
    def clip_box_nums(self, return_dict, keep_only_valid=True, num_clip_to=10):
        N_objs = len(return_dict['boxes_valid_list'])
        N_objs_valid = sum(return_dict['boxes_valid_list'])

        valid_idxes = [x for x in range(len(return_dict['boxes_valid_list'])) if return_dict['boxes_valid_list'][x]==True]
        invalid_idxes = [x for x in range(len(return_dict['boxes_valid_list'])) if return_dict['boxes_valid_list'][x]==False]
        # rearranged_indexes = valid_idxes + invalid_idxes

        assert keep_only_valid
        if N_objs_valid <= num_clip_to:
            s_idxes = valid_idxes
        else:
            # s_idxes = random.sample(range(N_objs), min(num_clip_to, N_objs))
            s_idxes = random.sample(valid_idxes, min(num_clip_to, N_objs_valid))
        N_objs_keep = len(s_idxes)
        return_dict['boxes_valid_list'] = [return_dict['boxes_valid_list'][x] for x in s_idxes]
        if 'mesh' in self.opt.cfg.DATA.data_read_list:
            return_dict['gt_obj_path_alignedNew_normalized_list'] = [return_dict['gt_obj_path_alignedNew_normalized_list'][x] for x in s_idxes]
            return_dict['gt_obj_path_alignedNew_original_list'] = [return_dict['gt_obj_path_alignedNew_original_list'][x] for x in s_idxes]

        for key in return_dict['boxes_batch']:
            if key in ['mask', 'random_id', 'cat_name']: # lists
                return_dict['boxes_batch'][key] = [return_dict['boxes_batch'][key][x] for x in s_idxes]
            elif key == 'g_feature': # subsample the relation features; g_feature are not symmetrical!
                assert return_dict['boxes_batch']['g_feature'].shape[0] == N_objs**2
                g_feature_channels = return_dict['boxes_batch']['g_feature'].shape[-1]
                g_feature_3d = return_dict['boxes_batch']['g_feature'].view(N_objs, N_objs, g_feature_channels)
                s_idxes_tensor = torch.tensor(s_idxes).flatten().long()
                xx, yy = torch.meshgrid(s_idxes_tensor, s_idxes_tensor)
                g_feature_3d_sampled = g_feature_3d[xx.long(), yy.long(), :]
                return_dict['boxes_batch']['g_feature'] = g_feature_3d_sampled.view(N_objs_keep**2, g_feature_channels)
            else: # arrays / tensors
                assert return_dict['boxes_batch'][key].shape[0] == N_objs
                return_dict['boxes_batch'][key] = return_dict['boxes_batch'][key][s_idxes]


        return return_dict

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
        elif key in ['frame_info', 'boxes_valid_list', 'emitter2wall_assign_info_list', 'emitters_obj_list', 'gt_layout_RAW', 'cell_info_grid', 'image_index', \
                'gt_obj_path_alignedNew_normalized_list', 'gt_obj_path_alignedNew_original_list', \
                'detectron_sample_dict', 'detectron_sample_dict']:
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            try:
                collated_batch[key] = default_collate([elem[key] for elem in batch])
            except e:
                print('[!!!!] Type error in collate_fn_OR: ', key, e)

    if 'boxes_batch' in batch[0]:
        interval_list = [elem['boxes_batch']['patch'].shape[0] for elem in batch]
        collated_batch['obj_split'] = torch.tensor([[sum(interval_list[:i]), sum(interval_list[:i+1])] for i in range(len(interval_list))])

    # boxes_valid_list = [item for sublist in collated_batch['boxes_valid_list'] for item in sublist]
    # boxes_valid_nums = [sum(x) for x in collated_batch['boxes_valid_list']]
    # boxes_total_nums = [len(x) for x in collated_batch['boxes_valid_list']]
    # if sum(boxes_valid_list)==0:
    #     print(boxes_valid_nums, '/', boxes_total_nums, red(sum(boxes_valid_list)), '/', len(boxes_valid_list), boxes_valid_list)
    # else:
    #     print(boxes_valid_nums, '/', boxes_total_nums, sum(boxes_valid_list), '/', len(boxes_valid_list), boxes_valid_list)

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
