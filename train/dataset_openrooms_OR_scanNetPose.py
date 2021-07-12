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
# import math
from detectron2.structures import BoxMode
from detectron2.data.dataset_mapper import DatasetMapper

from utils.utils_total3D.utils_OR_vis_labels import RGB_to_01
from utils.utils_total3D.utils_others import Relation_Config, OR4XCLASSES_dict, OR4XCLASSES_not_detect_mapping_ids_dict, OR4X_mapping_catInt_to_RGB
from detectron2.data import build_detection_test_loader,DatasetCatalog, MetadataCatalog

from utils.utils_scannet import read_ExtM_from_txt, read_img
import utils.utils_nvidia.mdataloader.m_preprocess as m_preprocess
import PIL
import torchvision.transforms as tfv_transform

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

def make_dataset(opt, split='train', data_root=None, data_list=None, logger=None):
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
        item = (image_name, label_name)
        image_label_list.append(item)
        meta_split_scene_name_frame_id_list.append((line_split[2].split('/')[0], line_split[0], int(line_split[1])))

    logger.info("==> Checking image&label pair [%s] list done! %d frames."%(split, len(image_label_list)))
    # print(image_label_list[:5])
    if opt.cfg.DATASET.first != -1:
        return image_label_list[:opt.cfg.DATASET.first], meta_split_scene_name_frame_id_list[:opt.cfg.DATASET.first]
    else:
        return image_label_list, meta_split_scene_name_frame_id_list


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
        assert self.split in ['train', 'val', 'test']
        self.task = self.split if task is None else task
        self.if_for_training = if_for_training
        self.data_root = self.opt.cfg.DATASET.dataset_path
        split_to_list = {'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt'}
        data_list = os.path.join(self.cfg.PATH.root, self.cfg.DATASET.dataset_list)
        data_list = os.path.join(data_list, split_to_list[split])
        self.data_list, self.meta_split_scene_name_frame_id_list = make_dataset(opt, split, self.data_root, data_list, logger=self.logger)
        assert len(self.data_list) == len(self.meta_split_scene_name_frame_id_list)
        if load_first != -1:
            self.data_list = self.data_list[:load_first] # [('/data/ruizhu/openrooms_mini-val/mainDiffLight_xml1/scene0509_00/im_1.hdr', '/data/ruizhu/openrooms_mini-val/main_xml1/scene0509_00/imsemLabel_1.npy'), ...
            self.meta_split_scene_name_frame_id_list = self.meta_split_scene_name_frame_id_list[:load_first] # [('mainDiffLight_xml1', 'scene0509_00', 1)

        logger.info(white_blue('%s-%s: total frames: %d'%(self.dataset_name, self.split, len(self.dataset_name))))

        self.OR = opt.cfg.MODEL_LAYOUT_EMITTER.data.OR
        self.valid_class_ids = RECON_3D_CLS_OR_dict[self.OR]

        self.cascadeLevel = cascadeLevel

        assert transforms_fixed is not None, 'OpenRooms: Need a transforms_fixed!'
        self.transforms_fixed = transforms_fixed
        self.transforms_resize = transforms_resize
        self.transforms_matseg = transforms_matseg
        self.transforms_semseg = transforms_semseg

        self.logger = logger
        # self.target_hw = (cfg.DATA.im_height, cfg.DATA.im_width) # if split in ['train', 'val', 'test'] else (args.test_h, args.test_w)
        self.im_width, self.im_height = self.cfg.DATA.im_width, self.cfg.DATA.im_height

        self.OR = self.cfg.MODEL_LAYOUT_EMITTER.data.OR
        self.OR_classes = OR4XCLASSES_dict[self.OR]

        if self.opt.cfg.MODEL_GMM.enable:
            self.to_gray = tfv_transform.Compose( [tfv_transform.Grayscale(),
                tfv_transform.ToTensor()])
            self.T_to_tensor = tfv_transform.ToTensor()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        hdr_image_path, semseg_label_path = self.data_list[index]
        meta_split, scene_name, frame_id = self.meta_split_scene_name_frame_id_list[index]
        assert frame_id > 0
        scene_total3d_path = Path(self.cfg.DATASET.layout_emitter_path) / meta_split / scene_name
        png_image_path = Path(self.opt.cfg.DATASET.png_path) / meta_split / scene_name / ('im_%d.png'%frame_id)

        frame_info = {'index': index, 'meta_split': meta_split, 'scene_name': scene_name, 'frame_id': frame_id, 'frame_key': '%s-%s-%d'%(meta_split, scene_name, frame_id), \
            'scene_total3d_path': scene_total3d_path, 'png_image_path': png_image_path}
        batch_dict = {'image_index': index, 'frame_info': frame_info}

        if_load_immask = self.opt.cfg.DATA.load_brdf_gt and not self.opt.cfg.DATA.if_load_png_not_hdr and (not self.opt.cfg.DATASET.if_no_gt)

        if if_load_immask:
            seg_path = hdr_image_path.replace('im_', 'immask_').replace('hdr', 'png').replace('DiffMat', '')
            # Read segmentation
            seg = 0.5 * (self.loadImage(seg_path ) + 1)[0:1, :, :]
            semantics_path = hdr_image_path.replace('DiffMat', '').replace('DiffLight', '')
            mask_path = semantics_path.replace('im_', 'imcadmatobj_').replace('hdr', 'dat')
            mask = self.loadBinary(mask_path, channels = 3, dtype=np.int32, if_resize=True).squeeze() # [h, w, 3]
        else:
            seg = np.ones((1, self.im_height, self.im_width), dtype=np.float32)
            mask_path = None
            mask = None

        hdr_scale = 1.

        assert self.opt.cfg.DATA.if_load_png_not_hdr
        if png_image_path.exists():
            # png_image_path.unlink()
            pass
        else:
            self.convert_write_png(hdr_image_path, seg, png_image_path)

        # Read PNG image
        image = Image.open(str(png_image_path))
        im_RGB_uint8 = np.array(image)
        im_RGB_uint8 = cv2.resize(im_RGB_uint8, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )

        image_transformed_fixed = self.transforms_fixed(im_RGB_uint8)
        im_trainval_RGB = self.transforms_resize(im_RGB_uint8) # not necessarily \in [0., 1.] [!!!!]
        # print(type(im_trainval_RGB), torch.max(im_trainval_RGB), torch.min(im_trainval_RGB), torch.mean(im_trainval_RGB))
        im_SDR_RGB = im_RGB_uint8.astype(np.float32) / 255.
        im_trainval = im_SDR_RGB # [240, 320, 3], np.ndarray

        batch_dict.update({'image_path': str(png_image_path)})

        if self.opt.cfg.DATA.if_also_load_next_frame:
            png_image_next_path = Path(self.opt.cfg.DATASET.png_path) / meta_split / scene_name / ('im_%d.png'%(frame_id+1))
            if not png_image_next_path.exists():
                return self.__getitem__((index+1)%len(self.data_list))
            image_next = Image.open(str(png_image_next_path))
            im_RGB_uint8_next = np.array(image_next)
            im_RGB_uint8_next = cv2.resize(im_RGB_uint8_next, (self.im_width, self.im_height), interpolation = cv2.INTER_AREA )
            im_SDR_RGB_next = im_RGB_uint8_next.astype(np.float32) / 255.
            batch_dict.update({'im_SDR_RGB_next': im_SDR_RGB_next})

        
        # image_transformed_fixed: normalized, not augmented [only needed in semseg]

        # im_trainval: normalized, augmented; HDR (same as im_trainval in png case) -> for input to network

        # im_trainval_RGB: normalized, augmented; LDR
        # im_SDR_RGB: normalized, NOT augmented; LDR
        # im_RGB_uint8: im_SDR_RGB -> 255

        # print('------', image_transformed_fixed.shape, im_trainval.shape, im_trainval_RGB.shape, im_SDR_RGB.shape, im_RGB_uint8.shape, )
        # png: ------ torch.Size([3, 240, 320]) (240, 320, 3) torch.Size([3, 240, 320]) (240, 320, 3) (240, 320, 3)
        # hdr: ------ torch.Size([3, 240, 320]) (3, 240, 320) (3, 240, 320) (3, 240, 320) (240, 320, 3)
        batch_dict.update({'hdr_scale': hdr_scale, 'image_transformed_fixed': image_transformed_fixed, 'im_trainval': torch.from_numpy(im_trainval), 'im_trainval_RGB': im_trainval_RGB, 'im_SDR_RGB': im_SDR_RGB, 'im_RGB_uint8': im_RGB_uint8})

        # ====== BRDF =====
        # image_path = batch_dict['image_path']
        if self.opt.cfg.DATA.load_brdf_gt and (not self.opt.cfg.DATASET.if_no_gt):
            batch_dict_brdf = self.load_brdf_lighting(hdr_image_path, if_load_immask, mask_path, mask, seg, hdr_scale, frame_info)
            batch_dict.update(batch_dict_brdf)

        if self.opt.cfg.MODEL_GMM.enable:
            self.load_scannet_compatible(batch_dict, frame_info)

        return batch_dict

    def convert_write_png(self, hdr_image_path, seg, png_image_path):
        # Read HDR image
        im_ori = self.loadHdr(hdr_image_path)
        # == no random scaling for inference
        im_SDR_fixedscale, _ = self.scaleHdr(im_ori, seg, forced_fixed_scale=True)
        im_SDR_RGB = np.clip(im_SDR_fixedscale**(1.0/2.2), 0., 1.)
        im_RGB_uint8 = (255. * im_SDR_RGB).transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(im_RGB_uint8).save(png_image_path)
        print(yellow('>>> Saved png file to %s'%png_image_path))

    def load_scannet_compatible(self, batch_dict, frame_info):
        meta_split, scene_name, frame_id = frame_info['meta_split'], frame_info['scene_name'], frame_info['frame_id']
        if self.opt.cfg.DATA.load_cam_pose:
            # if loading OR cam.txt files: need to inverse of def computeCameraEx() /home/ruizhu/Documents/Projects/Total3DUnderstanding/utils_OR/DatasetCreation/sampleCameraPoseFromScanNet.py
            cam_txt_path = Path(self.opt.cfg.DATASET.dataset_path) / meta_split / scene_name / ('pose_%d.txt'%frame_id)
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

        scene_dict = self.read_scene(frame_info=frame_info)
        cam_K = scene_dict['camera']['K'] # [[577.8708   0.     320.    ], [  0.     577.8708 240.    ], [  0.       0.       1.    ]]
        cam_K_ratio_W = cam_K[0][2] / (self.im_width/2.)
        cam_K_ratio_H = cam_K[1][2] / (self.im_height/2.)
        assert cam_K_ratio_W == cam_K_ratio_H
        cam_K_scaled = np.vstack([cam_K[:2, :] / cam_K_ratio_W, cam_K[2:3, :]])
        batch_dict.update({'cam_K_scaled_GMM': cam_K_scaled})
        # print(cam_K_scaled)


    def load_brdf_lighting(self, hdr_image_path, if_load_immask, mask_path, mask, seg, hdr_scale, frame_info):
        batch_dict_brdf = {}
        # Get paths for BRDF params
        if 'al' in self.cfg.DATA.data_read_list:
            albedo_path = hdr_image_path.replace('im_', 'imbaseColor_').replace('rgbe', 'png').replace('hdr', 'png')
            # Read albedo
            albedo = self.loadImage(albedo_path, isGama = False)
            albedo = (0.5 * (albedo + 1) ) ** 2.2
            batch_dict_brdf.update({'albedo': torch.from_numpy(albedo)})

        if 'no' in self.cfg.DATA.data_read_list:
            normal_path = hdr_image_path.replace('im_', 'imnormal_').replace('rgbe', 'png').replace('hdr', 'png')
            # normalize the normal vector so that it will be unit length
            normal = self.loadImage(normal_path )
            normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]
            batch_dict_brdf.update({'normal': torch.from_numpy(normal),})

        if 'ro' in self.cfg.DATA.data_read_list:
            rough_path = hdr_image_path.replace('im_', 'imroughness_').replace('rgbe', 'png').replace('hdr', 'png')
            # Read roughness
            rough = self.loadImage(rough_path )[0:1, :, :]
            batch_dict_brdf.update({'rough': torch.from_numpy(rough),})

        if 'de' in self.cfg.DATA.data_read_list or 'de' in self.cfg.DATA.data_read_list:
            depth_path = hdr_image_path.replace('im_', 'imdepth_').replace('rgbe', 'dat').replace('hdr', 'dat')
            # Read depth
            depth = self.loadBinary(depth_path)
            batch_dict_brdf.update({'depth': torch.from_numpy(depth),})
            if self.opt.cfg.DATA.if_also_load_next_frame:
                frame_id = frame_info['frame_id']
                depth_path_next = depth_path.replace('%d.dat'%frame_id, '%d.dat'%(frame_id+1))
                depth_next = self.loadBinary(depth_path_next)
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

        if if_load_immask:
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

        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale 

    def loadBinary(self, imName, channels = 1, dtype=np.float32, if_resize=True):
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
            depth = np.asarray(struct.unpack(decode_char * channels * height * width, dBuffer), dtype=dtype)
            depth = depth.reshape([height, width, channels] )
            if if_resize:
                # print(self.im_width, self.im_height, width, height)
                if dtype == np.float32:
                    depth = cv2.resize(depth, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA )
                elif dtype == np.int32:
                    depth = cv2.resize(depth.astype(np.float32), (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
                    depth = depth.astype(np.int32)

            depth = np.squeeze(depth)

        return depth[np.newaxis, :, :]

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
            except TypeError:
                print('[!!!!] Type error in collate_fn_OR: ', key)

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
