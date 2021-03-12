import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
import cv2
from skimage.measure import block_reduce
#import h5py
import scipy.ndimage as ndimage
from tqdm import tqdm
import torch
import timeit
import xml.etree.ElementTree as ET
import os

class BatchLoader(Dataset):
    def __init__(self, dataRoot, matDataRoot=None, dirs=['main_xml', 'main_xml1',
                                                         'mainDiffLight_xml', 'mainDiffLight_xml1',
                                                         'mainDiffMat_xml', 'mainDiffMat_xml1'],
                 imHeight=240, imWidth=320,
                 phase='TRAIN', split='train', rseed=None, mode='classifier'):

        print('Initializing dataloader ...')

        if phase.upper() == 'TRAIN':
            self.sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase.upper() == 'TEST':
            self.sceneFile = osp.join(dataRoot, 'test.txt')
        else:
            print('Unrecognized phase for data loader')
            assert(False)

        self.split = split
        assert self.split in ['train', 'val', 'test']

        self.mode = mode
        assert self.mode in ['cs', 'w', 'wn', 'w+n']

        # Read Material List
        matG2File = osp.join(dataRoot, 'matIdGlobal2.txt')
        matG2Dict = {}
        matG2ScaleDict = {}
        with open(matG2File, 'r') as f:
            for line in f.readlines():
                if 'Material__' not in line:
                    continue
                matName, r, g, b, rough, mId = line.strip().split(' ')
                matG2Dict[int(mId)] = matName
                matG2ScaleDict[int(mId)] = '%s_%s_%s_%s' % (r, g, b, rough)
        self.matG2Dict = matG2Dict
        self.matG2ScaleDict = matG2ScaleDict

        # Read Scene List
        with open(self.sceneFile, 'r') as fIn:
            sceneList = fIn.readlines()
        sceneList = [x.strip() for x in sceneList]
        num_scenes = len(sceneList)
        if phase.upper() == 'TRAIN':
            train_count = int(num_scenes * 0.95)
            val_count = num_scenes - train_count
            if self.split == 'train':
                sceneList = sceneList[:-val_count]
            if self.split == 'val':
                sceneList = sceneList[-val_count:]
        print('Scene num for split %s: %d; total scenes: %d' %
            (self.split, len(sceneList), num_scenes))
        

        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()
        self.matDataRoot = matDataRoot

        shapeList = []
        for d in dirs:
            shapeList = shapeList + \
                [osp.join(dataRoot, d, x) for x in sceneList]
        shapeList = sorted(shapeList)
        print('Shape Num: %d' % len(shapeList))

        self.imList = []
        self.matNameList = []
        self.matIdG2List = []
        if self.mode == 'cs':
            self.matIdG1List = []
            self.matScaleList = []

        listFile = '%s_list_%s.txt' % (self.split, self.mode)
        if osp.exists(listFile):
            print('Loading training list from %s' % listFile)
            with open(listFile, 'r') as f:
                for line in f.readlines():
                    if self.mode == 'cs':
                        im, matIdG2, matName, matScale, matIdG1 = line.strip().split(' ')
                        self.matIdG1List.append(int(matIdG1))
                        self.matScaleList.append(matScale)                        
                    else:
                        im, matIdG2, matName = line.strip().split(' ')
                    self.imList.append(im)
                    self.matIdG2List.append(int(matIdG2))
                    self.matNameList.append(matName)

        else:
            print('Creating training list ...')
            for shape in tqdm(shapeList):
                imNames = sorted(glob.glob(osp.join(shape, 'im_*.hdr')))
                for im in imNames:
                    matG2IdFile = im.replace('im_', 'immatPartGlobal2Ids_').replace('hdr', 'npy')
                    if osp.exists(matG2IdFile):
                        matG2Ids = list(np.load(matG2IdFile) )
                    else:
                        # Material Masks for parts
                        matG2IdMap = np.load(im.replace(
                            'im_', 'immatPartGlobal2_').replace('hdr', 'npy'))
                        matG2Ids = sorted(list(np.unique(matG2IdMap) ) )
                        if matG2Ids[0] == 0:
                            matG2Ids = matG2Ids[1:]
                        np.save(matG2IdFile, np.array(matG2Ids))

                    idNum = len(matG2Ids)
                    self.imList += ([im] * idNum)
                    self.matIdG2List += matG2Ids
                    matNameCurr = [self.matG2Dict[matG2Id] for matG2Id in matG2Ids]
                    self.matNameList += matNameCurr
                    #self.matNameList.append(self.matG2Dict[matG2Id])
                    if self.mode == 'cs':
                        matScaleCurr = [self.matG2ScaleDict[matG2Id] for matG2Id in matG2Ids]
                        self.matScaleList += matScaleCurr
                        #self.matScaleList.append(self.matG2ScaleDict[matG2Id])
                        matG1IdFile = im.replace('im_', 'immatPartGlobal1Ids_').replace('hdr', 'npy')
                        if osp.exists(matG1IdFile):
                            matG1Ids = list(np.load(matG1IdFile))
                            self.matIdG1List += matG1Ids
                        else:
                            matG1IdMap = np.load(im.replace(
                                'im_', 'immatPartGlobal1_').replace('hdr', 'npy'))
                            matG1Ids = []
                            if (osp.exists(matG2IdFile)): # in case matG2IdMap hasn't assigned yet.
                                matG2IdMap = np.load(im.replace(
                                    'im_', 'immatPartGlobal2_').replace('hdr', 'npy'))
                            for matG2Id in matG2Ids:
                                matMask = matG2IdMap == matG2Id
                                matG1Id = np.unique(matG1IdMap.flatten()[matMask.flatten()])[0]
                                matG1Ids.append(matG1Id)
                            self.matIdG1List += matG1Ids
                            assert len(self.matIdG2List) == len(self.matIdG1List)
                            np.save(matG1IdFile, np.array(matG1Ids))
                        with open(listFile, 'a') as f:
                            for idx in range(idNum):
                                f.writelines('%s %d %s %s %d\n' % (im, matG2Ids[idx], matNameCurr[idx], matScaleCurr[idx], matG1Ids[idx]) )
                    else:
                        with open(listFile, 'a') as f:
                            for idx in range(idNum):
                                f.writelines('%s %d %s\n' % (im, matG2Ids[idx], matNameCurr[idx]) )

            print('Pre-processed list is saved at %s' % listFile)                

        print('Data Pair Num: %d' % len(self.imList))

        # Input other than RGB
        self.depthList = [x.replace('im_', 'imdepth_').replace(
            'hdr', 'dat') for x in self.imList]
        self.depthList = [x.replace('DiffLight', '') for x in self.depthList]
        self.depthList = [x.replace('DiffMat', '') for x in self.depthList]

        self.segList = [x.replace('im_', 'immask_').replace(
            'hdr', 'png') for x in self.imList]
        self.segList = [x.replace('DiffMat', '') for x in self.segList]

        # # Material Masks
        #self.maskList = [x.replace('im_', 'immatPart_').replace('hdr', 'dat') for x in self.imList ]
        #self.maskList = [x.replace('DiffLight', '').replace('DiffMat', '') for x in self.maskList ]
        #self.maskG1List = [x.replace('immatPart_', 'immatPartGlobal1_').replace('dat', 'npy') for x in self.maskList ]
        self.maskG2List = [x.replace('im_', 'immatPartGlobal2_').replace(
            'hdr', 'npy') for x in self.imList]
        #self.maskG2List = [x.replace('DiffLight', '') for x in self.maskG2List]

        if phase.upper() != 'TRAIN':
            # Scene-based BRDF parameter
            self.albedoList = [x.replace('im_', 'imbaseColor_').replace(
                'hdr', 'png') for x in self.imList]

            self.normalList = [x.replace('im_', 'imnormal_').replace(
                'hdr', 'png') for x in self.imList]
            self.normalList = [x.replace('DiffLight', '') for x in self.normalList]

            self.roughList = [x.replace('im_', 'imroughness_').replace(
                'hdr', 'png') for x in self.imList]

        # Permute the image list
        self.count = len(self.imList)
        self.perm = list(range(self.count))

        if rseed is not None:
            random.seed(0)
        if self.split != 'val':
            random.shuffle(self.perm)

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, ind):
        startTime = timeit.default_timer()
        # Read segmentation
        seg = 0.5 * \
            (self.loadImage(self.segList[self.perm[ind]]) + 1)[0:1, :, :] # ~ 0.007 ms
        # Read Image
        im = self.loadHdr(self.imList[self.perm[ind]]) # ~ 0.02 ms
        # Random scale the image
        im, scale = self.scaleHdr(im, seg) # ~ 0.02 ms, mostly sorting operation
        t11 = timeit.default_timer() - startTime
        # Read depth
        depth = self.loadBinary(self.depthList[self.perm[ind]]) # ~ 0.02 ms
        depth = 1 / np.clip(depth + 1, 1e-6, 10)
        t12 = timeit.default_timer() - startTime
        # Read obj mask
        matG2IdMap = self.loadNPY(self.maskG2List[self.perm[ind]])
        matIdG2 = self.matIdG2List[self.perm[ind]]
        matMask = (matG2IdMap == matIdG2)[np.newaxis, :, :]
        t13 = timeit.default_timer() - startTime
        matName = self.matNameList[self.perm[ind]]

        batchDict = {
            'im': im,
            'depth': depth,
            'matMask': matMask,
            'matName': matName,
            'imPath': self.imList[self.perm[ind]]
        }
        t1 = timeit.default_timer() - startTime
        if self.phase.upper() != 'TRAIN':
            # Read Scene-based BRDF
            albedo = self.loadImage(self.albedoList[self.perm[ind]], isGama=False)
            albedo = (0.5 * (albedo + 1)) ** 2.2
            # # normalize the normal vector so that it will be unit length
            normal = self.loadImage(self.normalList[self.perm[ind]])
            normal = normal / \
                np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5)
                        )[np.newaxis, :]

            rough = self.loadImage(self.roughList[self.perm[ind]])[0:1, :, :]
            batchDict['albedo'] = albedo
            batchDict['normal'] = normal
            batchDict['rough'] = rough
        t2 = timeit.default_timer() - startTime
        # Read material output
        if self.mode == 'cs':
            matIdG1 = self.matIdG1List[self.perm[ind]] - 1z
            scales = self.matScaleList[self.perm[ind]].split('_')
            matScale = np.array([float(scales[0]), float(
                scales[1]), float(scales[2]), float(scales[3])]).astype(np.float32)
            batchDict['matLabel'] = matIdG1
            batchDict['matScale'] = matScale
            t3 = timeit.default_timer() - startTime
        elif self.mode == 'w':
            #matStyle = torch.load(
            #    osp.join(self.matDataRoot, matName, 'optim_latent.pt')).detach().cpu().numpy().squeeze()  # 512
            matStyle = np.load(osp.join(self.matDataRoot, matName, 'optim_latent.npy') )
            batchDict['matStyle'] = matStyle
            t3 = timeit.default_timer() - startTime
        elif self.mode == 'w+':
            matStyle = torch.load(
                osp.join(self.matDataRoot, matName, 'optim_latent.pt')).detach().cpu().numpy().squeeze()  # 512 x 14
            batchDict['matStyle'] = matStyle
        elif self.mode == 'w+n':
            matStyle = torch.load(
                osp.join(self.matDataRoot, matName, 'optim_latent.pt'))
            matNoise = torch.load(
                osp.join(self.matDataRoot, matName, 'optim_noise.pt'))
            batchDict['matStyle'] = matStyle
            batchDict['matNoise'] = matNoise
        tArr = np.array([t1, t2-t1, t3-t2])
        tpArr = np.array([t1 / t3, (t2-t1) / t3, (t3-t2) / t3])
        t1Arr = np.array([t11, t12-t11, t13-t12])
        batchDict['tArr'] = tArr
        batchDict['tpArr'] = tpArr
        batchDict['t1Arr'] = t1Arr
        return batchDict

    def get_map_aggre_map(self, objMask):
        cad_map = objMask[:, :, 0]
        mat_idx_map = objMask[:, :, 1]

        mat_aggre_map = np.zeros_like(cad_map)
        cad_ids = np.unique(cad_map)
        num_mats = 0
        for cad_id in cad_ids:
            cad_mask = cad_map == cad_id
            mat_map_cad = mat_idx_map[cad_mask]
            mat_idxs = np.unique(mat_map_cad)
            mat_aggre_map[cad_mask] = mat_idx_map[cad_mask] + num_mats
            num_mats = num_mats + max(mat_idxs) + 1

        return mat_aggre_map

    def loadImage(self, imName, isGama=False):
        if not(osp.isfile(imName)):
            print(imName)
            assert(False)

        #im = Image.open(imName)
        im = cv2.imread(imName, -1)
        #t0 = timeit.default_timer()
        im = cv2.resize(im, (self.imWidth, self.imHeight),
                        interpolation=cv2.INTER_AREA)
        #im = im.resize([self.imWidth, self.imHeight], Image.ANTIALIAS)
        #print('ImageResize hdr image: %.4f' % (timeit.default_timer() - t0) )

        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :]
        return im

    def loadHdr(self, imName):
        if not(osp.isfile(imName)):
            print(imName)
            assert(False)
        im = cv2.imread(imName, -1) # 0.02~0.03 ms, bottleneck
        if im is None:
            print(imName)
            assert(False)
        im = cv2.resize(im, (self.imWidth, self.imHeight),
                        interpolation=cv2.INTER_AREA)
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :]

        return im

    def scaleHdr(self, hdr, seg):
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.phase.upper() == 'TRAIN':
            scale = (0.95 - 0.1 * np.random.random()) / np.clip(
                intensityArr[int(0.95 * self.imWidth * self.imHeight * 3)], 0.1, None)
        elif self.phase.upper() == 'TEST':
            scale = (0.95 - 0.05) / np.clip(
                intensityArr[int(0.95 * self.imWidth * self.imHeight * 3)], 0.1, None)
        hdr = scale * hdr
        
        return np.clip(hdr, 0, 1), scale

    def loadBinary(self, imName, channels=1, dtype=np.float32, if_resize=True):
        assert dtype in [
            np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
        if not(osp.isfile(imName)):
            print(imName)
            assert(False)
        with open(imName, 'rb') as fIn:
            hBuffer = fIn.read(4)
            height = struct.unpack('i', hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack('i', wBuffer)[0]
            dBuffer = fIn.read(4 * channels * width * height)
            if dtype == np.float32:
                decode_char = 'f'
            elif dtype == np.int32:
                decode_char = 'i'
            depth = np.asarray(struct.unpack(
                decode_char * channels * height * width, dBuffer), dtype=dtype) # 0.02 ms ~ 0.1 ms, slow
            depth = depth.reshape([height, width, channels])

            if if_resize:
                #t0 = timeit.default_timer()
                # print(self.imWidth, self.imHeight, width, height)
                if dtype == np.float32:
                    depth = cv2.resize(
                        depth, (self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA)
                    #print('Resize float binary: %.4f' % (timeit.default_timer() - t0) )
                elif dtype == np.int32:
                    depth = cv2.resize(depth.astype(
                        np.float32), (self.imWidth, self.imHeight), interpolation=cv2.INTER_NEAREST)
                    depth = depth.astype(np.int32)
                    #print('Resize int32 binary: %.4f' % (timeit.default_timer() - t0) )

            depth = np.squeeze(depth)

        return depth[np.newaxis, :, :]

    def loadNPY(self, imName, dtype=np.int32, if_resize=True):
        depth = np.load(imName)
        if if_resize:
            # print(self.imWidth, self.imHeight, width, height)
            #t0 = timeit.default_timer()
            if dtype == np.float32:
                depth = cv2.resize(
                    depth, (self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA)
                #print('Resize float npy: %.4f' % (timeit.default_timer() - t0) )
            elif dtype == np.int32:
                depth = cv2.resize(depth.astype(
                    np.float32), (self.imWidth, self.imHeight), interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.int32)
                #print('Resize int32 npy: %.4f' % (timeit.default_timer() - t0) )

        depth = np.squeeze(depth)

        return depth

    # def loadH5(self, imName):
    #     try:
    #         hf = h5py.File(imName, 'r')
    #         im = np.array(hf.get('data'))
    #         return im
    #     except:
    #         return None

    def loadEnvmap(self, envName):
        if not osp.isfile(envName):
            env = np.zeros([3, self.envRow, self.envCol,
                            self.envHeight, self.envWidth], dtype=np.float32)
            envInd = np.zeros([1, 1, 1], dtype=np.float32)
            print('Warning: the envmap %s does not exist.' % envName)
            return env, envInd
        else:
            envHeightOrig, envWidthOrig = 16, 32
            assert((envHeightOrig / self.envHeight)
                   == (envWidthOrig / self.envWidth))
            assert(envHeightOrig % self.envHeight == 0)

            env = cv2.imread(envName, -1)

            if not env is None:
                env = env.reshape(self.envRow, envHeightOrig, self.envCol,
                                  envWidthOrig, 3)
                env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3]))

                scale = envHeightOrig / self.envHeight
                if scale > 1:
                    env = block_reduce(env, block_size=(
                        1, 1, 1, 2, 2), func=np.mean)

                envInd = np.ones([1, 1, 1], dtype=np.float32)
                return env, envInd
            else:
                env = np.zeros([3, self.envRow, self.envCol,
                                self.envHeight, self.envWidth], dtype=np.float32)
                envInd = np.zeros([1, 1, 1], dtype=np.float32)
                print('Warning: the envmap %s does not exist.' % envName)
                return env, envInd

            return env, envInd

# class EvalSceneLoader(Dataset):
#     def __init__(self, sceneRoot, modelListRoot=None, 
#                  imHeight=240, imWidth=320,
#                  phase='TEST', rseed=None, rgbMode = 'im', mode = 'cs', maskMode = 'default', vId = '*'):

#         ###  batch size is always 1, one batch contains 1 rgb + 1 depth + k part masks
#         ###  # of batchs = # of samples per scene
#         print('Initializing dataloader ...')

#         self.imHeight = imHeight
#         self.imWidth = imWidth
#         self.rgbMode = rgbMode
#         self.phase = phase
#         self.mode = mode
#         self.maskMode = maskMode # 'default' or 'mmap'
#         self.maskModeStr = maskMode if maskMode == 'mmap' else ''

#         self.imList = []
#         self.vId = vId

#         if rgbMode == 'im':
#             imNames = sorted(glob.glob(osp.join(sceneRoot, 'im_*.png')))
#             imStr = 'im'
#         elif rgbMode == 'imscannet':
#             #imNames = sorted(glob.glob(osp.join(sceneRoot, 'imscannet_*.png')))
#             imNames = sorted(glob.glob(osp.join(sceneRoot, 'imscannet_%s.png' % self.vId )))
#             imStr = 'imscannet'
#         else:
#             assert(False)
#         self.imList += imNames

#         # Input other than RGB
#         self.depthList = [x.replace('%s_' % rgbMode, 'imdepth_').replace(
#             'png', 'dat') for x in self.imList]
#         self.depthList = [x.replace('DiffLight', '') for x in self.depthList]
#         self.depthList = [x.replace('DiffMat', '') for x in self.depthList]

#         # self.segList = [x.replace('im_', 'immask_').replace(
#         #     'hdr', 'png') for x in self.imList]
#         # self.segList = [x.replace('DiffMat', '') for x in self.segList]

#         # Read Model List
#         modelListFile = open(modelListRoot, 'r')
#         self.modelStrList = []
#         for i, line in enumerate(modelListFile.readlines()):
#             if i == 0:
#                 continue
#             item = line.strip().split(' ')
#             self.modelStrList.append('%s_%s' % (item[0], item[1]))

#         self.matSegDirList = []
#         # Fetch part information from xml file
#         self.xmlFile = osp.join(sceneRoot, 'main.xml')
#         partIdDictPath = osp.join(sceneRoot, 'partIdDict.txt')
#         objBaseDir = osp.dirname(modelListRoot) # assume layoutMesh located as same as models.txt
#         if not osp.exists(partIdDictPath): # save part mask separately
#             print('%s not exist!' % partIdDictPath)
#             objPartDict = self.getPartDictFromXml(self.xmlFile, objBaseDir) 
#             #print(objPartDict)
#             # {objName: objParts} / xxx_yyy: [xxx_yyy_part0, xxx_yyy_part1]
#             self.partIdDict = self.createPartIdDict(objPartDict, partIdDictPath) # {xxx_yyy_part0: 0, xxx_yyy_part1: 1, xxx_zzz_part0: 2}
#             for im in imNames:
#                 matPart = im.replace('%s_' % imStr, 'immatPart_').replace('png', 'dat')
#                 matSegDir = matPart.replace('.dat', '')
#                 os.system('mkdir -p %s' % matSegDir)
#                 matSegFileList = []
#                 matMap = self.loadBinary(matPart, channels = 3, dtype=np.int32, if_resize=False).squeeze() # [h, w, 3]
#                 #h, w, c = matMap.shape
#                 cadIdMap = matMap[:,:,0]
#                 matIdMap = matMap[:,:,1]
#                 cadIds = np.unique(cadIdMap)
#                 for cadId in cadIds:
#                     cadMask = cadIdMap == cadId
#                     matIds = np.unique(matIdMap[cadMask])
#                     if cadId == 0:
#                         continue
#                     for matId in matIds: # from 1
#                         objStrId = self.modelStrList[cadId-1]
#                         parts = objPartDict[objStrId]
#                         bsdfStrId = parts[matId-1] # xxx_yyy_part0
#                         partId = self.partIdDict[bsdfStrId]
#                         # save separate mat seg
#                         matMask = matIdMap == matId
#                         matGlobalMask = cadMask * matMask
#                         matSegFile = osp.join(matSegDir, '%d.png' % partId)
#                         matSegFileList.append(matSegFile)
#                         Image.fromarray(matGlobalMask.astype(np.uint8) * 255).save(matSegFile)
#                 self.matSegDirList.append(matSegFileList)
#         else:
#             print('%s exist!' % partIdDictPath)
#             self.partIdDict = self.loadPartIdDict(partIdDictPath)
#             for im in imNames:
#                 matPart = im.replace('%s_' % imStr, 'immatPart_').replace('png', 'dat')
#                 matSegDir = matPart.replace('.dat', '')
#                 matSegFileList = glob.glob(osp.join(matSegDir, '*.png') )
#                 self.matSegDirList.append(matSegFileList)

#         print('Num of View in this scene: %d' % len(self.imList))
#         ############

#         # Permute the image list
#         self.count = len(self.imList)
#         self.perm = list(range(self.count))

#         # if rseed is not None:
#         #     random.seed(0)
#         # random.shuffle(self.perm)

#     def getPartDictFromXml(self, xmlFile, objBaseDir):
#         if not osp.exists(xmlFile):
#             assert(False)
#         tree = ET.parse(xmlFile)
#         root = tree.getroot()
#         objPartDict = {} # {objName: objParts} / xxx_yyy: [xxx_yyy_part0, xxx_yyy_part1]
#         objPathList = [] # layoutMesh/sceenXXXX_XX/XXX.obj
#         for child in root: # Collect All object and part info per scene
#             if child.tag == 'shape':
#                 shapeStrId = child.attrib['id'] # cadcatID_objID_object
#                 for child2 in child:
#                     if child2.tag == 'string' and child2.attrib['name'] == 'filename':
#                         objPath = child2.attrib['value'].replace('../../../../../', '')
#                         break
#                 cond1 = 'aligned_light.obj' in objPath
#                 cond2 = 'container.obj' in objPath
#                 cond3 = objPath in objPathList
#                 if cond1 or cond2 or cond3:
#                     continue
#                 objPathFull = osp.join(objBaseDir, objPath)
#                 if not osp.exists(objPathFull):
#                     print('ObjFile %s not exist!' % objPathFull)
#                     assert(False)
#                 objParts = []
#                 with open(objPathFull, 'r') as obj:
#                     for line in obj.readlines():
#                         if 'usemtl' not in line:
#                             continue
#                         objPartName = line.strip().split(' ')[1]
#                         objParts.append(objPartName)
#                 objPartDict[shapeStrId.replace('_object', '')] = objParts
#                 objPathList.append(objPath)
        
#         return objPartDict # {objName: objParts} / xxx_yyy: [xxx_yyy_part0, xxx_yyy_part1]

#     def createPartIdDict(self, objPartDict, savePath):
#         keyList = sorted(objPartDict.keys())
#         partId = 0
#         partIdDict = {}
#         with open(savePath, 'w') as f:
#             for k in keyList:
#                 parts = objPartDict[k]
#                 for p in parts:
#                     partIdDict[p] = partId
#                     f.writelines('%s %d\n' % (p, partId))
#                     partId += 1
#         return partIdDict

#     def loadPartIdDict(self, loadPath):
#         partIdDict = {}
#         with open(loadPath, 'r') as f:
#             for line in f.readlines():
#                 partName, partId = line.strip().split(' ')
#                 partIdDict[partName] = int(partId)
#         return partIdDict
    
#     def __len__(self):
#         return len(self.perm)

#     def __getitem__(self, ind):
#         # if self.rgbMode == 'im':
#         #     # Read Image
#         #     im = self.loadHdr(self.imList[self.perm[ind]]) # ~ 0.02 ms
#         #     seg = np.ones_like(im)[0:1, :, :]
#         #     # Random scale the image
#         #     im, _ = self.scaleHdr(im, seg) # ~ 0.02 ms, mostly sorting operation
#         # elif self.rgbMode == 'imscannet':
#         im = self.loadImage(self.imList[self.perm[ind]], isGama=True)

#         # Read depth
#         depth = self.loadBinary(self.depthList[self.perm[ind]]) # ~ 0.02 ms
#         depth = 1 / np.clip(depth + 1, 1e-6, 10)

#         # Read seg mask
#         matSegFileList = self.matSegDirList[self.perm[ind]]
#         numSegs = len(matSegFileList)
#         segs = []
#         partIds = []
#         for matSegFile in matSegFileList:
#             if self.maskMode == 'mmap':
#                 basename = osp.basename(matSegFile)
#                 matSegFileLoad = matSegFile.replace(basename, 'mapped_sn_mask_%s' % basename).replace('immatPart_', 'immatPartMapped_')
#                 if not osp.exists(matSegFileLoad):
#                     matSegFileLoad = matSegFile
#             else:
#                 matSegFileLoad = matSegFile
#             seg = 0.5 * \
#                 (self.loadImage(matSegFileLoad) + 1)[0:1, :, :] # ~ 0.007 ms
#             segs.append(seg)
#             partId = osp.basename(matSegFile).split('.')[0]
#             partIds.append(partId)
        
#         segs = np.stack(segs, axis=0) # nSegs x 1 x h x w
#         ims = np.tile(im, (numSegs, 1, 1, 1)) # nSegs x 3 x h x w
#         depths = np.tile(depth, (numSegs, 1, 1, 1)) # nSegs x 1 x h x w
#         # partIds: list of nSegs items

#         viewId = self.imList[self.perm[ind]].split('_')[-1].split('.')[0]
#         xmlOutPath = osp.join(osp.dirname(self.imList[self.perm[ind]]), '%s%s%s_%s.xml' % (self.rgbMode, self.mode, self.maskModeStr, viewId) )

#         batchDict = {
#             'im': ims,
#             'depth': depths,
#             'matMask': segs,
#             'partId': partIds,
#             'matPredDir': osp.join(osp.dirname(self.imList[self.perm[ind]]), '%s%s%smatPred_%s' % (self.rgbMode, self.mode, self.maskModeStr, viewId) ),
#             'xmlOutPath': xmlOutPath
#         }
#         return batchDict

#     def loadImage(self, imName, isGama=False):
#         if not(osp.isfile(imName)):
#             print(imName)
#             assert(False)

#         #im = Image.open(imName)
#         im = cv2.imread(imName, -1)
#         #t0 = timeit.default_timer()
#         im = cv2.resize(im, (self.imWidth, self.imHeight),
#                         interpolation=cv2.INTER_AREA)
#         #im = im.resize([self.imWidth, self.imHeight], Image.ANTIALIAS)
#         #print('ImageResize hdr image: %.4f' % (timeit.default_timer() - t0) )

#         im = np.asarray(im, dtype=np.float32)
#         if isGama:
#             im = (im / 255.0) ** 2.2
#             im = 2 * im - 1
#         else:
#             im = (im - 127.5) / 127.5
#         if len(im.shape) == 2:
#             im = im[..., np.newaxis]
#         im = np.transpose(im, [2, 0, 1])
#         im = im[::-1, :, :]
#         return im

#     def loadHdr(self, imName):
#         if not(osp.isfile(imName)):
#             print(imName)
#             assert(False)
#         im = cv2.imread(imName, -1) # 0.02~0.03 ms, bottleneck
#         if im is None:
#             print(imName)
#             assert(False)
#         im = cv2.resize(im, (self.imWidth, self.imHeight),
#                         interpolation=cv2.INTER_AREA)
#         im = np.asarray(im, dtype=np.float32)
#         im = np.transpose(im, [2, 0, 1])
#         im = im[::-1, :, :]

#         return im

#     def scaleHdr(self, hdr, seg):
#         intensityArr = (hdr * seg).flatten()
#         intensityArr.sort()
#         if self.phase.upper() == 'TRAIN':
#             scale = (0.95 - 0.1 * np.random.random()) / np.clip(
#                 intensityArr[int(0.95 * self.imWidth * self.imHeight * 3)], 0.1, None)
#         elif self.phase.upper() == 'TEST':
#             scale = (0.95 - 0.05) / np.clip(
#                 intensityArr[int(0.95 * self.imWidth * self.imHeight * 3)], 0.1, None)
#         hdr = scale * hdr
        
#         return np.clip(hdr, 0, 1), scale

#     def loadBinary(self, imName, channels=1, dtype=np.float32, if_resize=True):
#         assert dtype in [
#             np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
#         if not(osp.isfile(imName)):
#             print(imName)
#             assert(False)
#         with open(imName, 'rb') as fIn:
#             hBuffer = fIn.read(4)
#             height = struct.unpack('i', hBuffer)[0]
#             wBuffer = fIn.read(4)
#             width = struct.unpack('i', wBuffer)[0]
#             dBuffer = fIn.read(4 * channels * width * height)
#             if dtype == np.float32:
#                 decode_char = 'f'
#             elif dtype == np.int32:
#                 decode_char = 'i'
#             depth = np.asarray(struct.unpack(
#                 decode_char * channels * height * width, dBuffer), dtype=dtype) # 0.02 ms ~ 0.1 ms, slow
#             depth = depth.reshape([height, width, channels])

#             if if_resize:
#                 #t0 = timeit.default_timer()
#                 # print(self.imWidth, self.imHeight, width, height)
#                 if dtype == np.float32:
#                     depth = cv2.resize(
#                         depth, (self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA)
#                     #print('Resize float binary: %.4f' % (timeit.default_timer() - t0) )
#                 elif dtype == np.int32:
#                     depth = cv2.resize(depth.astype(
#                         np.float32), (self.imWidth, self.imHeight), interpolation=cv2.INTER_NEAREST)
#                     depth = depth.astype(np.int32)
#                     #print('Resize int32 binary: %.4f' % (timeit.default_timer() - t0) )

#             depth = np.squeeze(depth)

#         return depth[np.newaxis, :, :]

# class MatWLoader(Dataset):
    def __init__(self, matDataRoot):
        self.matDataRoot = matDataRoot
        self.matNameList = glob.glob(osp.join(matDataRoot, 'Material*') )

        self.count = len(self.matNameList)
        self.perm = list(range(self.count))

        random.seed(0)
        random.shuffle(self.perm)

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, ind):
        matName = self.matNameList[self.perm[ind]]
        matStyle = np.load(osp.join(matName, 'optim_latent.npy') )
        matNoisePath = osp.join(matName, 'optim_noise.pt')
        batchDict = {
            'matName': matName,
            'matStyle': matStyle,
            'matNoisePath': matNoisePath
        }
        return batchDict