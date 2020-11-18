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
from utils.utils_misc import *

def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            image_name = os.path.join(data_root, line_split[0])
            if len(line_split) != 1:
                label_name = os.path.join(data_root, line_split[1])
                # raise (RuntimeError("Image list file read line error : " + line + "\n"))
            else:
                label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
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
    print("Checking image&label pair {} list done!".format(split))
    # print(image_label_list[:5])
    return image_label_list


class openrooms(data.Dataset):
    def __init__(self, _data_root, transform, opt, data_list=None, logger=None, 
            split='train', load_first = -1, rseed = None, 
            cascadeLevel = 0,
            isLight = False, isAllLight = False,
            envHeight = 8, envWidth = 16, envRow = 120, envCol = 160, 
            SGNum = 12):

        self.opt = opt
        self.cfg = self.opt.cfg
        self.logger = logger
        self.dataset_name = self.cfg.DATASET.dataset_name
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.data_root = self.opt.cfg.DATASET.dataset_path
        split_to_list = {'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt'}
        data_list = os.path.join(self.cfg.PATH.root, self.cfg.DATASET.dataset_list)
        data_list = os.path.join(data_list, split_to_list[split])
        self.data_list = make_dataset(split, self.data_root, data_list)
        if load_first != -1:
            self.data_list = self.data_list[:load_first]
        logger.info(white_blue('%s-%s: total frames: %d'%(self.dataset_name, self.split, len(self.dataset_name))))

        self.transform = transform
        self.logger = logger
        # self.target_hw = (cfg.DATA.im_height, cfg.DATA.im_width) # if split in ['train', 'val', 'test'] else (args.test_h, args.test_w)
        self.im_width, self.im_height = self.cfg.DATA.im_width, self.cfg.DATA.im_height

        if self.opt.cfg.MODEL_SEMSEG.enable:
            self.semseg_colors = np.loadtxt(self.cfg.PATH.semseg_colors_path).astype('uint8')
            self.semseg_names = [line.rstrip('\n') for line in open(self.cfg.PATH.semseg_names_path)]
            assert len(self.semseg_colors) == len(self.semseg_names)

        self.cascadeLevel = cascadeLevel
        self.isLight = isLight

    def __len__(self):
        return len(self.data_list)

        
    def __getitem__(self, index):

        image_path, semseg_label_path = self.data_list[index]

        # Get paths for BRDF params
        albedo_path = image_path.replace('im_', 'imbaseColor_').replace('hdr', 'png') 
        normal_path = image_path.replace('im_', 'imnormal_').replace('hdr', 'png').replace('DiffLight', '')
        rough_path = image_path.replace('im_', 'imroughness_').replace('hdr', 'png')
        depth_path = image_path.replace('im_', 'imdepth_').replace('hdr', 'dat').replace('DiffLight', '').replace('DiffMat', '')
        mask_path = image_path.replace('im_', 'immatPart_').replace('hdr', 'dat')
        seg_path = image_path.replace('im_', 'immask_').replace('hdr', 'png').replace('DiffMat', '')

        if self.cascadeLevel == 0:
            if self.isLight:
                env_path = image_path.replace('im_', 'imenv_')
        else:
            if self.isLight:
                env_path = image_path.replace('im_', 'imenv_')
                envPre_path = image_path.replace('im_', 'imenv_').replace('.hdr', '_%d.h5'  % (self.cascadeLevel -1) )
            
            albedoPre_path = image_path.replace('im_', 'imbaseColor_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )
            normalPre_path = image_path.replace('im_', 'imnormal_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )
            roughPre_path = image_path.replace('im_', 'imroughness_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )
            depthPre_path = image_path.replace('im_', 'imdepth_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )

            diffusePre_path = image_path.replace('im_', 'imdiffuse_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )
            specularPre_path = image_path.replace('im_', 'imspecular_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )

        # Read segmentation
        seg = 0.5 * (self.loadImage(seg_path ) + 1)[0:1, :, :]
        segArea = np.logical_and(seg > 0.49, seg < 0.51 ).astype(np.float32 )
        segEnv = (seg < 0.1).astype(np.float32 )
        segObj = (seg > 0.9) 

        if self.isLight:
            segObj = segObj.squeeze()
            segObj = ndimage.binary_erosion(segObj, structure=np.ones((7, 7) ),
                    border_value=1)
            segObj = segObj[np.newaxis, :, :]

        segObj = segObj.astype(np.float32 )

        # Read Image
        hdr_file = image_path
        im_ori = self.loadHdr(hdr_file)
        # Random scale the image
        im, scale = self.scaleHdr(im_ori, seg)

        assert self.transform is not None
        im_not_hdr = np.clip(im**(1.0/2.2), 0., 1.)
        im_uint8 = (255. * im_not_hdr).transpose(1, 2, 0).astype(np.uint8)
        image = Image.fromarray(im_uint8)
        image_transformed = self.transform(image)

        # Read albedo
        albedo = self.loadImage(albedo_path, isGama = False)
        albedo = (0.5 * (albedo + 1) ) ** 2.2

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage(normal_path )
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]

        # Read roughness
        rough = self.loadImage(rough_path )[0:1, :, :]

        # Read depth
        depth = self.loadBinary(depth_path)

        # >>>> Rui: Read obj mask
        mask = self.loadBinary(mask_path, channels = 3, dtype=np.int32, if_resize=True).squeeze() # [h, w, 3]
        mat_aggre_map, num_mat_masks = self.get_map_aggre_map(mask) # 0 for invalid region
        mat_aggre_map = mat_aggre_map[:, :, np.newaxis]

        mat_aggre_map_reindex = np.zeros_like(mat_aggre_map)
        mat_aggre_map_reindex[mat_aggre_map==0] = self.opt.invalid_index
        mat_aggre_map_reindex[mat_aggre_map!=0] = mat_aggre_map[mat_aggre_map!=0] - 1


        # >>>> Rui: Convert to Plane-paper compatible formats
        h, w, _ = mat_aggre_map.shape
        gt_segmentation = mat_aggre_map
        segmentation = np.zeros([50, h, w], dtype=np.uint8)
        for i in range(num_mat_masks+1):
            if i == 0:
                # deal with backgroud
                seg = gt_segmentation == 0
                segmentation[num_mat_masks, :, :] = seg.reshape(h, w)
            else:
                seg = gt_segmentation == i
                segmentation[i-1, :, :] = seg.reshape(h, w)
        # <<<<

        if self.isLight == True:
            envmaps, envmapsInd = self.loadEnvmap(env_path )
            envmaps = envmaps * scale 
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

        batch_dict = {
                'albedo': torch.from_numpy(albedo),
                'normal': torch.from_numpy(normal),
                'rough': torch.from_numpy(rough),
                'depth': torch.from_numpy(depth),
                'mask': torch.from_numpy(mask), 
                'maskPath': mask_path, 
                'segArea': torch.from_numpy(segArea),
                'segEnv': torch.from_numpy(segEnv),
                'segObj': torch.from_numpy(segObj),
                'im': torch.from_numpy(im),
                'object_type_seg': torch.from_numpy(seg), 
                'imPath': image_path, 
                'mat_aggre_map': torch.from_numpy(mat_aggre_map), 
                'mat_aggre_map_reindex': torch.from_numpy(mat_aggre_map_reindex), # gt_seg
                'num_mat_masks': num_mat_masks,  
                'mat_notlight_mask': torch.from_numpy(mat_aggre_map!=0).float(),
                'instance': torch.ByteTensor(segmentation), 
                'semantic': 1 - torch.FloatTensor(segmentation[num_mat_masks, :, :]).unsqueeze(0),
                }
        # if self.transform is not None and not self.opt.if_hdr:
        batch_dict.update({'image_transformed': image_transformed, 'im_not_hdr': im_not_hdr, 'im_uint8': im_uint8})

        if self.isLight:
            batch_dict['envmaps'] = envmaps
            batch_dict['envmapsInd'] = envmapsInd

            if self.cascadeLevel > 0:
                batch_dict['envmapsPre'] = envmapsPre

        if self.cascadeLevel > 0:
            batch_dict['albedoPre'] = albedoPre
            batch_dict['normalPre'] = normalPre
            batch_dict['roughPre'] = roughPre
            batch_dict['depthPre'] = depthPre

            batch_dict['diffusePre'] = diffusePre
            batch_dict['specularPre'] = specularPre
            
        if self.opt.cfg.DATA.load_semseg_gt:
            semseg_label = np.load(semseg_label_path)
            # semseg_label[semseg_label==0] = 31
            semseg_label = cv2.resize(semseg_label, (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
            batch_dict.update({'semseg_label': torch.from_numpy(semseg_label).long()})

        return batch_dict

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

    def loadImage(self, imName, isGama = False):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )

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

    def scaleHdr(self, hdr, seg):
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.split == 'train':
            scale = (0.95 - 0.1 * np.random.random() )  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
        else:
            scale = (0.95 - 0.05)  / np.clip(intensityArr[int(0.95 * self.im_width * self.im_height * 3) ], 0.1, None)
        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale 

    def loadBinary(self, imName, channels = 1, dtype=np.float32, if_resize=True):
        assert dtype in [np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )
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
        print('>>>>loadEnvmap', envName)
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
                    envWidthOrig, 3)
                env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3] ) )

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
                

            return env, envInd
