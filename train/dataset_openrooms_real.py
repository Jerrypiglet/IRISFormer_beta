import glob
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
import PIL


class openrooms_real(data.Dataset):
    def __init__(self, dataRoot, transform, opt, image_list, 
            imHeight = 240, imWidth = 320):
        
        self.dataset_name = 'openrooms-REAL'
        self.opt = opt
        self.split = 'test'

        if self.opt.cfg.MODEL_SEMSEG.enable:
            self.semseg_path = self.opt.semseg_configs.semseg_path_cluster if opt.if_cluster else self.opt.semseg_configs.semseg_path_local
            self.semseg_colors = np.loadtxt(self.semseg_path + opt.semseg_configs.colors_path).astype('uint8')
        
        with open(image_list, 'r') as fIn:
            self.imList = fIn.readlines() 
        self.imList = [x.strip() for x in self.imList]

        self.imHeight = imHeight
        self.imWidth = imWidth

        self.transform = transform

        self.imList = sorted(self.imList)
        print('Image Num: %d' % len(self.imList ), self.imList)

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, ind):

        # Read Image
        ldr_file = self.imList[ind]
        im_uint8 = np.array(Image.open(ldr_file).convert('RGB')).astype(np.uint8)
        im_uint8 = cv2.resize(im_uint8, dsize=(self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA)

        im_not_hdr = (im_uint8.astype(np.float) / 255.).transpose((2, 0, 1))
        image_ldr = np.clip(im_not_hdr ** 2.2, 0., 1.)

        image_transformed = self.transform(Image.fromarray(im_uint8))

        batchDict = {
                'albedo': torch.ones((3, self.imHeight, self.imWidth)).float(),
                'normal': torch.ones((3, self.imHeight, self.imWidth)).float(),
                'rough': torch.ones((1, self.imHeight, self.imWidth)).float(),
                'depth': torch.ones((1, self.imHeight, self.imWidth)).float(),
                'mask': torch.ones((self.imHeight, self.imWidth, 3)).float(), 
                'maskPath': '', 
                'segArea': torch.zeros((1, self.imHeight, self.imWidth)).float(),
                'segEnv': torch.zeros((1, self.imHeight, self.imWidth)).float(),
                'segObj': torch.ones((1, self.imHeight, self.imWidth)).float(),
                'im': torch.from_numpy(image_ldr).float(),
                'object_type_seg': torch.ones((self.imHeight, self.imWidth, 1)).float(), 
                'imPath': self.imList[ind], 
                'mat_aggre_map': torch.ones((self.imHeight, self.imWidth, 1)).float(), 
                'mat_aggre_map_reindex': torch.ones((self.imHeight, self.imWidth, 1)).float(), # gt_seg
                'num_mat_masks': 0,  
                'mat_notlight_mask': torch.ones((self.imHeight, self.imWidth, 1)).float(),
                'instance': torch.ones((50, self.imHeight, self.imWidth)).byte(), 
                'semantic': torch.zeros((1, self.imHeight, self.imWidth)).float(),
                }
        # if self.transform is not None and not self.opt.if_hdr:
        batchDict.update({'image_transformed': image_transformed, 'im_not_hdr': im_not_hdr, 'im_uint8': im_uint8})

        if self.opt.cfg.MODEL_BRDF.enable_semseg_decoder:
            batchDict.update({'semseg_label': torch.ones((self.imHeight, self.imWidth)).long()})

        return batchDict