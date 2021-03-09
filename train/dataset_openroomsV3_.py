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
from pathlib import Path
import pickle
import math

HEIGHT_PATCH = 256
WIDTH_PATCH = 256
from utils.utils_total3D.utils_OR_vis_labels import RGB_to_01
from utils.utils_total3D.utils_others import Relation_Config, OR4XCLASSES_dict, OR4XCLASSES_not_detect_mapping_ids_dict, OR4X_mapping_catInt_to_RGB
OR = 'OR45'
classes = OR4XCLASSES_dict[OR]

rel_cfg = Relation_Config()
d_model = int(rel_cfg.d_g/4)

data_transforms_crop = T.Compose([
    T.Resize((280, 280)),
    T.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_nocrop = T.Compose([
    T.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_nocrop_depth = T.Compose([
    T.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    T.ToTensor(),
])

data_transforms_crop_nonormalize = T.Compose([
    T.Resize((280, 280)),
    T.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
    T.ToTensor(),
])

data_transforms_nocrop_nonormalize = T.Compose([
    T.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    T.ToTensor(),
])


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
    def __init__(self, opt, data_list=None, logger=basic_logger(), transforms_fixed=None, transforms=None, transforms_matseg=None, 
            split='train', load_first = -1, rseed = None, 
            cascadeLevel = 0,
            # is_light = False, is_all_light = False,
            envHeight = 8, envWidth = 16, envRow = 120, envCol = 160, 
            SGNum = 12):

        if logger is None:
            logger = basic_logger()

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
        self.data_list, self.meta_split_scene_name_frame_id_list = make_dataset(split, self.data_root, data_list, logger=self.logger)
        assert len(self.data_list) == len(self.meta_split_scene_name_frame_id_list)
        if load_first != -1:
            self.data_list = self.data_list[:load_first]
            self.meta_split_scene_name_frame_id_list = self.meta_split_scene_name_frame_id_list[:load_first]
        logger.info(white_blue('%s-%s: total frames: %d'%(self.dataset_name, self.split, len(self.dataset_name))))

        self.cascadeLevel = cascadeLevel
        # self.is_all_light = self.opt.cfg.MODEL_BRDF.is_all_light
        
        # if self.is_all_light:
        #     logger.info('Filtering data_list with is_all_light=True...')
        #     num_before = len(self.data_list)
        #     self.data_list = [(item[0], item[1]) if os.path.isfile(item[0].replace('im_', 'imenv_')) for item in self.data_list]
        #     logger.info('Filtering done. Before %d, after %d.'%(num_before, len(self.data_list))
  
        assert transforms_fixed is not None, 'OpenRooms: Need a transforms_fixed!'
        self.transforms_fixed = transforms_fixed
        self.transforms = transforms
        self.transforms_matseg = transforms_matseg
        self.logger = logger
        # self.target_hw = (cfg.DATA.im_height, cfg.DATA.im_width) # if split in ['train', 'val', 'test'] else (args.test_h, args.test_w)
        self.im_width, self.im_height = self.cfg.DATA.im_width, self.cfg.DATA.im_height

        if self.opt.cfg.MODEL_SEMSEG.enable:
            self.semseg_colors = np.loadtxt(self.cfg.PATH.semseg_colors_path).astype('uint8')
            self.semseg_names = [line.rstrip('\n') for line in open(self.cfg.PATH.semseg_names_path)]
            assert len(self.semseg_colors) == len(self.semseg_names)
            
        # ====== layout, obj, emitters =====
        if self.opt.cfg.DATA.load_layout_emitter_gt:
            self.OR = self.cfg.MODEL_LAYOUT_EMITTER.OR
            self.grid_size = self.cfg.MODEL_LAYOUT_EMITTER.grid_size
            self.OR_classes = OR4XCLASSES_dict[self.OR]
            self.PNG_data_root = Path('/newfoundland2/ruizhu/siggraphasia20dataset/layout_labels_V4-ORfull/') if not opt.if_cluster else self.data_root
            self.layout_emitter_im_width, self.layout_emitter_im_height = WIDTH_PATCH, HEIGHT_PATCH
            with open(Path(self.cfg.PATH.total3D_colors_path) / OR4X_mapping_catInt_to_RGB['light'], 'rb') as f:
                self.OR_mapping_catInt_to_RGB = pickle.load(f)[self.OR]

        # ====== per-pixel lighting =====
        if self.opt.cfg.MODEL_LIGHT.enable:
            self.envWidth = envWidth
            self.envHeight = envHeight
            self.envRow = envRow
            self.envCol = envCol
            self.SGNum = SGNum

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index):
        
        image_path, semseg_label_path = self.data_list[index]
        seg_path = image_path.replace('im_', 'immask_').replace('hdr', 'png').replace('DiffMat', '')
        # Read segmentation
        seg = 0.5 * (self.loadImage(seg_path ) + 1)[0:1, :, :]
        semantics_path = image_path.replace('DiffMat', '').replace('DiffMat', '').replace('DiffLight', '')
        # mask_path = semantics_path.replace('im_', 'imcadmatobj_').replace('hdr', 'dat')
        mask_path = semantics_path.replace('im_', 'immatPart_').replace('hdr', 'dat')
        mask = self.loadBinary(mask_path, channels = 3, dtype=np.int32, if_resize=True).squeeze() # [h, w, 3]

        # Read Image
        hdr_file = image_path
        im_ori = self.loadHdr(hdr_file)
        # Random scale the image
        im, scale = self.scaleHdr(im_ori, seg)
        im_trainval_RGB = np.clip(im**(1.0/2.2), 0., 1.)

        # assert self.transforms_fixed is not None
        im_SDR_scale, _ = self.scaleHdr(im_ori, seg, forced_fixed_scale=True)
        im_SDR_RGB = np.clip(im_SDR_scale**(1.0/2.2), 0., 1.)
        im_RGB_uint8 = (255. * im_SDR_RGB).transpose(1, 2, 0).astype(np.uint8)
        # image = Image.fromarray(im_RGB_uint8)
        image_transformed_fixed = self.transforms_fixed(im_RGB_uint8)
        
        batch_dict = {'im': torch.from_numpy(im), 'image_path': image_path}
        batch_dict.update({'image_transformed_fixed': image_transformed_fixed, 'im_trainval_RGB': im_trainval_RGB, 'im_SDR_RGB': im_SDR_RGB, 'im_RGB_uint8': im_RGB_uint8})

        # ====== BRDF =====
        if self.opt.cfg.DATA.load_brdf_gt or len(self.opt.cfg.DATA.data_read_list) != 0:
            # Get paths for BRDF params
            if 'al' in self.cfg.MODEL_BRDF.enable_list:
                albedo_path = image_path.replace('im_', 'imbaseColor_').replace('hdr', 'png') 
                # Read albedo
                albedo = self.loadImage(albedo_path, isGama = False)
                albedo = (0.5 * (albedo + 1) ) ** 2.2
                batch_dict.update({'albedo': torch.from_numpy(albedo)})

            if 'no' in self.cfg.MODEL_BRDF.enable_list:
                normal_path = image_path.replace('im_', 'imnormal_').replace('hdr', 'png').replace('DiffLight', '')
                # normalize the normal vector so that it will be unit length
                normal = self.loadImage(normal_path )
                normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]
                batch_dict.update({'normal': torch.from_numpy(normal),})

            if 'ro' in self.cfg.MODEL_BRDF.enable_list:
                rough_path = image_path.replace('im_', 'imroughness_').replace('hdr', 'png')
                # Read roughness
                rough = self.loadImage(rough_path )[0:1, :, :]
                batch_dict.update({'rough': torch.from_numpy(rough),})

            if 'de' in self.cfg.MODEL_BRDF.enable_list or 'de' in self.cfg.DATA.data_read_list:
                depth_path = image_path.replace('im_', 'imdepth_').replace('hdr', 'dat').replace('DiffLight', '').replace('DiffMat', '')
                # Read depth
                depth = self.loadBinary(depth_path)
                batch_dict.update({'depth': torch.from_numpy(depth),})

            if self.cascadeLevel == 0:
                if self.opt.cfg.MODEL_LIGHT.enable:
                    env_path = image_path.replace('im_', 'imenv_')
            else:
                if self.opt.cfg.MODEL_LIGHT.enable:
                    env_path = image_path.replace('im_', 'imenv_')
                    envPre_path = image_path.replace('im_', 'imenv_').replace('.hdr', '_%d.h5'  % (self.cascadeLevel -1) )
                
                albedoPre_path = image_path.replace('im_', 'imbaseColor_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )
                normalPre_path = image_path.replace('im_', 'imnormal_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )
                roughPre_path = image_path.replace('im_', 'imroughness_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )
                depthPre_path = image_path.replace('im_', 'imdepth_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) )

                diffusePre_path = image_path.replace('im_', 'imdiffuse_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )
                specularPre_path = image_path.replace('im_', 'imspecular_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) )

            segArea = np.logical_and(seg > 0.49, seg < 0.51 ).astype(np.float32 )
            segEnv = (seg < 0.1).astype(np.float32 )
            segObj = (seg > 0.9) 

            if self.opt.cfg.MODEL_LIGHT.enable:
                segObj = segObj.squeeze()
                segObj = ndimage.binary_erosion(segObj, structure=np.ones((7, 7) ),
                        border_value=1)
                segObj = segObj[np.newaxis, :, :]

            segObj = segObj.astype(np.float32 )

            if self.opt.cfg.MODEL_LIGHT.enable:
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

            batch_dict.update({
                    'mask': torch.from_numpy(mask), 
                    'maskPath': mask_path, 
                    'segArea': torch.from_numpy(segArea),
                    'segEnv': torch.from_numpy(segEnv),
                    'segObj': torch.from_numpy(segObj),
                    'object_type_seg': torch.from_numpy(seg), 
                    })
            # if self.transform is not None and not self.opt.if_hdr:

            if self.opt.cfg.MODEL_LIGHT.enable:
                batch_dict['envmaps'] = envmaps
                batch_dict['envmapsInd'] = envmapsInd
                # print(envmaps.shape, envmapsInd.shape)

                if self.cascadeLevel > 0:
                    batch_dict['envmapsPre'] = envmapsPre

            if self.cascadeLevel > 0:
                batch_dict['albedoPre'] = albedoPre
                batch_dict['normalPre'] = normalPre
                batch_dict['roughPre'] = roughPre
                batch_dict['depthPre'] = depthPre

                batch_dict['diffusePre'] = diffusePre
                batch_dict['specularPre'] = specularPre
        
        # ====== semseg =====
        if self.opt.cfg.DATA.load_matseg_gt:
            mat_seg_dict = self.load_mat_seg(mask, im_RGB_uint8)
            batch_dict.update(mat_seg_dict)

        # ====== matseg =====
        if self.opt.cfg.DATA.load_semseg_gt:
            sem_seg_dict = self.load_mat_seg(im_RGB_uint8, semseg_label_path)
            batch_dict.update(sem_seg_dict)

        # ====== layout, obj, emitters =====
        if self.opt.cfg.DATA.load_layout_emitter_gt:
            layout_emitter_dict = self.load_layout_emitter_gt(self.meta_split_scene_name_frame_id_list[index])
            batch_dict.update(layout_emitter_dict)
        
        return batch_dict

    def load_sem_seg(self, im_RGB_uint8, semseg_label_path):
        semseg_label = np.load(semseg_label_path).astype(np.uint8)
        semseg_label = cv2.resize(semseg_label, (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
        # Transform images
        im_semseg_transformed_trainval, semseg_label = self.transforms(im_RGB_uint8, semseg_label) # augmented
        # semseg_label[semseg_label==0] = 31
        return {'semseg_label': semseg_label.long(), 'im_semseg_transformed_trainval': im_semseg_transformed_trainval}

    def load_mat_seg(self, mask, im_RGB_uint8):
        # >>>> Rui: Read obj mask
        mat_aggre_map, num_mat_masks = self.get_map_aggre_map(mask) # 0 for invalid region
        im_matseg_transformed_trainval, mat_aggre_map_transformed = self.transforms_matseg(im_RGB_uint8, mat_aggre_map.squeeze()) # augmented
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

    def load_layout_emitter_gt(self, meta_split_scene_name_frame_id):
        meta_split, scene_name, frame_id = meta_split_scene_name_frame_id
        pickle_path = str(Path(self.cfg.DATASET.layout_emitter_path) / meta_split / scene_name / ('layout_obj_%d.pkl'%frame_id))
        file_path = pickle_path.replace('.pkl', '_reindexed.pkl')
        with open(file_path, 'rb') as f:
            sequence = pickle.load(f)

        return_dict = {}

        png_image_path = Path(self.PNG_data_root) / meta_split / scene_name / ('im_%d.png'%frame_id)
        image = Image.open(str(png_image_path))
        image_np = np.array(image)

        camera = sequence['camera']
        boxes = sequence['boxes']
        n_objects = boxes['bdb2D_pos'].shape[0]

        boxes_valid_list = list(boxes['if_valid'] if 'if_valid' in boxes else [True]*n_objects)
        # if not self.config['has_invalid_objs']:
        #     assert all(boxes_valid_list), 'has invalid objs from file %s!'%file_path

        g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                      ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                      math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                      math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                     for id1, loc1 in enumerate(boxes['bdb2D_pos'])
                     for id2, loc2 in enumerate(boxes['bdb2D_pos'])]

        locs = [num for loc in g_feature for num in loc]

        pe = torch.zeros(len(locs), d_model)
        position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        boxes['g_feature'] = pe.view(n_objects * n_objects, rel_cfg.d_g)

        # encode class
        cls_codes = torch.zeros([len(boxes['size_cls']), len(self.OR_classes)])
        
        # if self.config['data']['dataset_super'] == 'OR': # OR: set cat_id==0 to invalid, (and [optionally] remap not-detect-cats to 0)
        assert len(boxes['size_cls']) == len(boxes_valid_list)
        for idx in range(len(boxes['size_cls'])):
            if boxes['size_cls'][idx] == 0:
                boxes_valid_list[idx] = False # set cat_id==0 to invalid
            if boxes['size_cls'][idx] in OR4XCLASSES_not_detect_mapping_ids_dict[self.OR]: # [optionally] remap not-detect-cats to 0
                boxes_valid_list[idx] = False

        cls_codes[range(len(boxes['size_cls'])), boxes['size_cls']] = 1
        boxes['size_cls'] = cls_codes

        layout = sequence['layout']

        # TODO: If the training error is consistently larger than the test error. We remove the crop and add more intermediate FC layers with no dropout.
        # TODO: Or FC layers with more hidden neurons, which ensures more neurons pass through the dropout layer, or with larger learning rate, longer
        # TODO: decay rate.
        data_transforms = data_transforms_crop if self.split == 'train' else data_transforms_nocrop
        # data_transforms_nonormalize = data_transforms_crop_nonormalize if self.mode=='train' else data_transforms_nocrop_nonormalize

        patch = []
        for bdb in boxes['bdb2D_pos']:
            img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
            # img_nonormalize = data_transforms_nonormalize(img)
            img = data_transforms(img)
            patch.append(img)
        boxes['patch'] = torch.stack(patch)
        image = data_transforms_nocrop(image)

        assert boxes['patch'].shape[0] == len(boxes_valid_list)

        return_dict.update({'image':image, 'image_np': image_np, 
            # 'rgb_img': torch.from_numpy(sequence['rgb_img']), 
            'rgb_img_path': str(sequence['rgb_img_path']), 'pickle_path': file_path, \
            'boxes_batch':boxes, 'camera':camera, 'layout':layout, 'sequence_id': sequence['sequence_id'], 
            'boxes_valid_list': boxes_valid_list})

        # if if_depth:
        #     return_dict.update({'depth': pil2tensor(depth).squeeze()})

        # --- reading emitters
        pickle_emitter2wall_assign_info_dict_path = pickle_path.replace('.pkl', '_emitters_assign_info_%dX%d_V3.pkl'%(self.grid_size, self.grid_size))
        if_emitter = Path(pickle_emitter2wall_assign_info_dict_path).exists()
        assert if_emitter, 'Cannot find' + pickle_emitter2wall_assign_info_dict_path
        if if_emitter:
            with open(pickle_emitter2wall_assign_info_dict_path, 'rb') as f:
                sequence_emitter2wall_assign_info_dict = pickle.load(f)
                # sequence_emitter2wall_assign_info[0].keys()
            emitter2wall_assign_info_list = sequence_emitter2wall_assign_info_dict['emitter2wall_assign_info_list']

            if self.cfg.MODEL_LAYOUT_EMITTER.wall_prob_or_cell_prob == 'wall_prob':
                wall_grid_prob = sequence_emitter2wall_assign_info_dict['wall_grid_prob']
                return_dict.update({'wall_grid_prob': torch.from_numpy(wall_grid_prob).float()})
            elif self.cfg.MODEL_LAYOUT_EMITTER.wall_prob_or_cell_prob == 'cell_prob':
                cell_prob_mean = sequence_emitter2wall_assign_info_dict['cell_prob_mean']
                return_dict.update({'cell_prob_mean': torch.from_numpy(cell_prob_mean).float()})

            emitters_obj_list = []

            pickle_emitters_path = pickle_path.replace('.pkl', '_emitters.pkl')
            with open(pickle_emitters_path, 'rb') as f:
                sequence_emitters = pickle.load(f)

            for x in range(sequence_emitters['boxes']['bdb3D'].shape[0]):
                obj_dict_new = {'obj_box_3d': sequence_emitters['boxes']['bdb3D'][x], 'cat_id': sequence_emitters['boxes']['size_cls'][x], \
                                'cat_name': classes[sequence_emitters['boxes']['size_cls'][x]], 'cat_color': RGB_to_01(self.OR_mapping_catInt_to_RGB[sequence_emitters['boxes']['size_cls'][x]])}
                emitters_obj_list.append(obj_dict_new)

            return_dict.update({'emitter2wall_assign_info_list': emitter2wall_assign_info_list, 'emitters_obj_list': emitters_obj_list, 'gt_layout_RAW': layout['bdb3D']})

        return return_dict

    
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
            self.logger.warning('File does not exist: ' + imName )
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

    def scaleHdr(self, hdr, seg, forced_fixed_scale=False):
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.split == 'train' and not forced_fixed_scale:
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
        elif key in ['boxes_valid_list', 'emitter2wall_assign_info_list', 'emitters_obj_list', 'gt_layout_RAW']:
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            # print(key)
            collated_batch[key] = default_collate([elem[key] for elem in batch])

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
