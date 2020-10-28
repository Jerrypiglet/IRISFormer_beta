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


class openrooms(data.Dataset):
    def __init__(self, dataRoot, transform, opt, dirs = ['main_xml', 'main_xml1',
        'mainDiffLight_xml', 'mainDiffLight_xml1', 
        'mainDiffMat_xml', 'mainDiffMat_xml1'], 
            imHeight = 240, imWidth = 320, 
            phase='TRAIN', split='train', rseed = None, cascadeLevel = 0,
            isLight = False, isAllLight = False,
            envHeight = 8, envWidth = 16, envRow = 120, envCol = 160, 
            SGNum = 12):
        
        if phase.upper() == 'TRAIN':
            self.sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase.upper() == 'TEST':
            self.sceneFile = osp.join(dataRoot, 'test.txt') 
        else:
            print('Unrecognized phase for data loader')
            assert(False )

        self.dataset_name = 'openrooms-' + dataRoot
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.opt = opt

        
        with open(self.sceneFile, 'r') as fIn:
            sceneList = fIn.readlines() 
        sceneList = [x.strip() for x in sceneList]

        if phase.upper() == 'TRAIN':
            num_scenes = len(sceneList)
            train_count = int(num_scenes * 0.95)
            val_count = num_scenes - train_count
            if self.split == 'train':
                sceneList = sceneList[:-val_count]
            if self.split == 'val':
                sceneList = sceneList[-val_count:]
        if 'mini' in self.dataset_name:
            sceneList *= 10
        print('Scene num for split %s: %d; total scenes: %d'%(self.split, len(sceneList), num_scenes))

        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()
        self.cascadeLevel = cascadeLevel
        self.isLight = isLight
        self.isAllLight = isAllLight
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.envRow = envRow
        self.envCol = envCol
        self.envWidth = envWidth 
        self.envHeight = envHeight
        self.SGNum = SGNum

        self.transform = transform

        shapeList = []
        for d in dirs:
            shapeList = shapeList + [osp.join(dataRoot, d, x) for x in sceneList ]
        shapeList = sorted(shapeList)
        print('Shape Num: %d' % len(shapeList ) )

        self.imList = []
        print('Gathering shapes...')
        for shape in tqdm(shapeList):
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.hdr') ) )
            self.imList = self.imList + imNames

        if isAllLight:
            self.imList = [x for x in self.imList if
                    osp.isfile(x.replace('im_', 'imenv_') ) ]
            if cascadeLevel > 0:
                self.imList = [x for x in self.imList if
                        osp.isfile(x.replace('im_',
                            'imenv_').replace('.hdr', '_%d.h5' %
                                (self.cascadeLevel - 1 )  ) ) ]


        print('Image Num: %d' % len(self.imList ) )

        # BRDF parameter
        self.albedoList = [x.replace('im_', 'imbaseColor_').replace('hdr', 'png') for x in self.imList ] 

        self.normalList = [x.replace('im_', 'imnormal_').replace('hdr', 'png') for x in self.imList ]
        self.normalList = [x.replace('DiffLight', '') for x in self.normalList ]

        self.roughList = [x.replace('im_', 'imroughness_').replace('hdr', 'png') for x in self.imList ]

        self.depthList = [x.replace('im_', 'imdepth_').replace('hdr', 'dat') for x in self.imList ]
        self.depthList = [x.replace('DiffLight', '') for x in self.depthList ]
        self.depthList = [x.replace('DiffMat', '') for x in self.depthList ]

        self.maskList = [x.replace('im_', 'immatPart_').replace('hdr', 'dat') for x in self.imList ]
        # self.maskList = [x.replace('DiffLight', '').replace('DiffMat', '') for x in self.maskList ]

        self.segList = [x.replace('im_', 'immask_').replace('hdr', 'png') for x in self.imList ]
        self.segList = [x.replace('DiffMat', '') for x in self.segList ]

        if self.cascadeLevel == 0:
            if self.isLight:
                self.envList = [x.replace('im_', 'imenv_') for x in self.imList ]
        else:
            if self.isLight:
                self.envList = [x.replace('im_', 'imenv_') for x in self.imList ]
                self.envPreList = [x.replace('im_', 'imenv_').replace('.hdr', '_%d.h5'  % (self.cascadeLevel -1) ) for x in self.imList ]
            
            self.albedoPreList = [x.replace('im_', 'imbaseColor_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) ) for x in self.imList ]
            self.normalPreList = [x.replace('im_', 'imnormal_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) ) for x in self.imList ]
            self.roughPreList = [x.replace('im_', 'imroughness_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) ) for x in self.imList ]
            self.depthPreList = [x.replace('im_', 'imdepth_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) ) for x in self.imList ]

            self.diffusePreList = [x.replace('im_', 'imdiffuse_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) ) for x in self.imList ]
            self.specularPreList = [x.replace('im_', 'imspecular_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) ) for x in self.imList ]

        # Permute the image list
        self.count = len(self.albedoList )
        self.perm = list(range(self.count ) )

        if rseed is not None:
            random.seed(0)
        random.shuffle(self.perm )

    def __len__(self):
        return len(self.perm )

    def __getitem__(self, ind):
        # Read segmentation
        seg = 0.5 * (self.loadImage(self.segList[self.perm[ind] ] ) + 1)[0:1, :, :]
        segArea = np.logical_and(seg > 0.49, seg < 0.51 ).astype(np.float32 )
        segEnv = (seg < 0.1).astype(np.float32 )
        segObj = (seg > 0.9) 

        # print(segArea.shape, segEnv.shape, segObj.shape)
        
        if self.isLight:
            segObj = segObj.squeeze()
            segObj = ndimage.binary_erosion(segObj, structure=np.ones((7, 7) ),
                    border_value=1)
            segObj = segObj[np.newaxis, :, :]

        segObj = segObj.astype(np.float32 )

        # print(segObj.shape)


        # Read Image
        hdr_file = self.imList[self.perm[ind] ]
        # if self.opt.if_hdr:
        im = self.loadHdr(hdr_file)
        # Random scale the image
        im, scale = self.scaleHdr(im, seg)

        # if not self.opt.if_hdr_input_mat_seg:
        assert self.transform is not None
        # sdr_file = self.imList[self.perm[ind] ].replace('.hdr', 'sdr')
        # im_uint8 = np.load(sdr_file+'.npy')
        im_not_hdr = np.clip(im**(1.0/2.2), 0., 1.)
        im_uint8 = (255. * im_not_hdr).transpose(1, 2, 0).astype(np.uint8)
        # np.save(sdr_file, im_uint8)
        # print(self.imList[self.perm[ind] ])
        # print('-', im_uint8.dtype, im_uint8.shape, np.amax(im_uint8), np.amin(im_uint8), np.median(im_uint8))
        image = Image.fromarray(im_uint8)
        image_transformed = self.transform(image)
        # print('----', image_transformed.dtype, torch.max(image_transformed), torch.min(image_transformed), torch.median(image_transformed))


        # Read albedo
        albedo = self.loadImage(self.albedoList[self.perm[ind] ], isGama = False)
        albedo = (0.5 * (albedo + 1) ) ** 2.2

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage(self.normalList[self.perm[ind] ] )
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]

        # Read roughness
        rough = self.loadImage(self.roughList[self.perm[ind] ] )[0:1, :, :]

        # Read depth
        depth = self.loadBinary(self.depthList[self.perm[ind] ])

        # >>>> Rui: Read obj mask
        mask = self.loadBinary(self.maskList[self.perm[ind] ], channels = 3, dtype=np.int32, if_resize=True).squeeze() # [h, w, 3]
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
            envmaps, envmapsInd = self.loadEnvmap(self.envList[self.perm[ind] ] )
            envmaps = envmaps * scale 
            if self.cascadeLevel > 0: 
                envmapsPre = self.loadH5(self.envPreList[self.perm[ind] ] ) 
                if envmapsPre is None:
                    print("Wrong envmap pred")
                    envmapsInd = envmapsInd * 0 
                    envmapsPre = np.zeros((84, 120, 160), dtype=np.float32 ) 

        if self.cascadeLevel > 0:
            # Read albedo
            albedoPre = self.loadH5(self.albedoPreList[self.perm[ind] ] )
            albedoPre = albedoPre / np.maximum(np.mean(albedoPre ), 1e-10) / 3

            # normalize the normal vector so that it will be unit length
            normalPre = self.loadH5(self.normalPreList[self.perm[ind] ] )
            normalPre = normalPre / np.sqrt(np.maximum(np.sum(normalPre * normalPre, axis=0), 1e-5) )[np.newaxis, :]
            normalPre = 0.5 * (normalPre + 1)

            # Read roughness
            roughPre = self.loadH5(self.roughPreList[self.perm[ind] ] )[0:1, :, :]
            roughPre = 0.5 * (roughPre + 1)

            # Read depth
            depthPre = self.loadH5(self.depthPreList[self.perm[ind] ] )
            depthPre = depthPre / np.maximum(np.mean(depthPre), 1e-10) / 3

            diffusePre = self.loadH5(self.diffusePreList[self.perm[ind] ] )
            diffusePre = diffusePre / max(diffusePre.max(), 1e-10)

            specularPre = self.loadH5(self.specularPreList[self.perm[ind] ] )
            specularPre = specularPre / max(specularPre.max(), 1e-10)

        batchDict = {
                'albedo': torch.from_numpy(albedo),
                'normal': torch.from_numpy(normal),
                'rough': torch.from_numpy(rough),
                'depth': torch.from_numpy(depth),
                'mask': torch.from_numpy(mask), 
                'maskPath': self.maskList[self.perm[ind] ], 
                'segArea': torch.from_numpy(segArea),
                'segEnv': torch.from_numpy(segEnv),
                'segObj': torch.from_numpy(segObj),
                'im': torch.from_numpy(im),
                'object_type_seg': torch.from_numpy(seg), 
                'imPath': self.imList[self.perm[ind] ], 
                'mat_aggre_map': torch.from_numpy(mat_aggre_map), 
                'mat_aggre_map_reindex': torch.from_numpy(mat_aggre_map_reindex), # gt_seg
                'num_mat_masks': num_mat_masks,  
                'mat_notlight_mask': torch.from_numpy(mat_aggre_map!=0).float(),
                'instance': torch.ByteTensor(segmentation), 
                'semantic': 1 - torch.FloatTensor(segmentation[num_mat_masks, :, :]).unsqueeze(0),
                }
        # if self.transform is not None and not self.opt.if_hdr:
        batchDict.update({'image_transformed': image_transformed, 'im_not_hdr': im_not_hdr})

        if self.isLight:
            batchDict['envmaps'] = envmaps
            batchDict['envmapsInd'] = envmapsInd

            if self.cascadeLevel > 0:
                batchDict['envmapsPre'] = envmapsPre

        if self.cascadeLevel > 0:
            batchDict['albedoPre'] = albedoPre
            batchDict['normalPre'] = normalPre
            batchDict['roughPre'] = roughPre
            batchDict['depthPre'] = depthPre

            batchDict['diffusePre'] = diffusePre
            batchDict['specularPre'] = specularPre

        return batchDict

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
        im = im.resize([self.imWidth, self.imHeight], Image.ANTIALIAS )

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
        im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :]
        return im

    def scaleHdr(self, hdr, seg):
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.phase.upper() == 'TRAIN':
            scale = (0.95 - 0.1 * np.random.random() )  / np.clip(intensityArr[int(0.95 * self.imWidth * self.imHeight * 3) ], 0.1, None)
        elif self.phase.upper() == 'TEST':
            scale = (0.95 - 0.05)  / np.clip(intensityArr[int(0.95 * self.imWidth * self.imHeight * 3) ], 0.1, None)
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
                # print(self.imWidth, self.imHeight, width, height)
                if dtype == np.float32:
                    depth = cv2.resize(depth, (self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA )
                elif dtype == np.int32:
                    depth = cv2.resize(depth.astype(np.float32), (self.imWidth, self.imHeight), interpolation=cv2.INTER_NEAREST)
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
