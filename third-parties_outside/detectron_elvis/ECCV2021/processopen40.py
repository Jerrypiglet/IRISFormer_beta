import glob 
import numpy as np 
import os.path as osp 
import os  
import struct  
import cv2  
import scipy.ndimage as ndimage

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
from skimage import measure
import h5py
import scipy.ndimage as ndimage
import pickle
from detectron2.structures import BoxMode
import os
import matplotlib.pyplot as plt
import torch

prefix='/eccv20dataset/elvis/labelTransform/'
def fromSegToInsSeg(labelName, matName, nyuMap ):
    semLabel = np.load(labelName )
    with open(matName, 'rb') as fIn:
        hBuffer = fIn.read(4)
        height = struct.unpack('i', hBuffer )[0]
        wBuffer = fIn.read(4)
        width = struct.unpack('i', wBuffer )[0] 
        matLabel = fIn.read(4 * 3 * width * height ) 
        matLabel = np.array(struct.unpack(
            'i' * height * width * 3, matLabel ), dtype=np.int32 ) 
        matLabel = matLabel.reshape(height, width,3  )
        insLabel = matLabel[:, :, 2] 

    minInsL = max(insLabel.min(), 1 )
    maxInsL = insLabel.max() 
    clses, masks = [], [] 
    for n in range(minInsL, maxInsL + 1):
        mask = (insLabel == n).astype(np.int32 ) 
        maskErode = ndimage.binary_erosion(
                mask, structure=np.ones( (5, 5) ) )
        if np.sum(maskErode ) == 0:
            continue 
        else:
            cls = -1
            maxNum = 0
            for n in range(0, 46 ):
                semMask = (semLabel == n).astype(np.int32 )
                num = np.sum(semMask * mask ) 
                if num > maxNum:
                    maxNum = num 
                    cls = n 
            assert(cls >= 0 )
            if cls == 31: # Window 
                maskX = np.sum(mask, axis=0 )
                xArr = np.where(maskX > 0 )[0] 
                xMin, xMax = xArr.min(), xArr.max() 

                maskY = np.sum(mask, axis=1 )
                yArr = np.where(maskY > 0 )[0]
                yMin, yMax = yArr.min(), yArr.max()  

                box = np.zeros(mask.shape, mask.dtype )
                box[yMin:yMax, xMin:xMax ] = 1
                box = box * (semLabel == 31)
                mask = np.clip(mask + box, 0, 1)

            
            masks.append(mask.astype(np.uint8 )[:, :, np.newaxis] )
            clses.append(cls ) 
    
    masks = np.concatenate(masks, axis=-1 )
    clses = np.array(clses ).astype(np.uint16 )[:, np.newaxis ]
    clses = nyuMap[clses ]

    return clses, masks 


class open40loader(Dataset):
    def __init__(self, dataRoot = '/siggraphasia20dataset/code/Routine/DatasetCreation',
            imHeight = 480, imWidth = 640,
            phase='TRAIN', rseed = None,
            isLight = True,
            isLightBox = True,
            envHeight = 8, envWidth = 16):
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()
        if self.phase=="TEST":
            dataRoot='/eccv20dataset/DatasetNew_test/'
        self.dataRoot=dataRoot
        self.isLight = isLight
        self.isLightBox = isLightBox

        self.envWidth = envWidth
        self.envHeight = envHeight
        self.outdirr='/eccv20dataset/elvis/outfileelvis/'
        # Load shapes
        shapeList = []
        dirs = ['main_xml', 'main_xml1']
        if phase.upper() == 'TRAIN':
            sceneFile = osp.join(dataRoot, 'train.txt')
            #self.outdir=dataRoot+'/train_data/'
            self.outdir=self.outdirr+'/train_data_open_40'
        elif phase.upper() == 'TEST':
            sceneFile = osp.join(dataRoot, 'test.txt')
            #self.outdir=dataRoot+'/test_data'
            self.outdir=self.outdirr+'/test_data_open_40'
        else:
            print('Wrong: unrecognizable phase')
            assert(False )
        with open(sceneFile, 'r') as fIn:
            sceneList = fIn.readlines()
        sceneList = [x.strip() for x in sceneList ]
        print(sceneFile)
        for d in dirs:
            shapeList = shapeList + [osp.join(dataRoot, d, x) for x in sceneList ]
        shapeList = sorted(shapeList )
        # ['./sample/main_xml/scene0001_00', './sample/main_xml1/scene0001_00']
        # all the sub-directory
        print('Shape Num: %d' % len(shapeList ) )

        self.imList = []
        for shape in shapeList:
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.hdr') ) )
            self.imList = self.imList + imNames
        self.imList=self.imList[:15]
        self.imDlList = [x.replace('main_', 'mainDiffLight_') for x in self.imList ]
        self.imDmList = [x.replace('main_', 'mainDiffMat_') for x in self.imList ]
        print('Image Num: %d' % (len(self.imList ) * 3) )
        self.segList = [x.replace('im_', 'immask_').replace('hdr', 'png') for x in self.imList ]
        self.segDlList = [x.replace('main_', 'mainDiffLight_') for x in self.segList ]
    def get_imlist(self):
        return self.imList,self.imDlList,self.imDmList


    def loadImage(self, imName, isGama = False):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )

        im = cv2.imread(imName )
        im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation =
                cv2.INTER_AREA )
        im = np.ascontiguousarray(im[:, :, ::-1] )

        im = np.asarray(im, dtype=np.float32 )
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1] )

        return im

    def loadSeg(self, name ):
        seg = 0.5 * (self.loadImage(name ) + 1)[0:1, :, :]
        segArea = np.logical_and(seg > 0.05, seg < 0.95 ).astype(np.float32 )
        segEnv = (seg < 0.05).astype(np.float32 )
        segObj = (seg > 0.95 )

        if self.isLight:
            segObj = segObj.squeeze()
            segObj = ndimage.binary_erosion(segObj, structure=np.ones((7, 7) ),
                    border_value=1)
            segObj = segObj[np.newaxis, :, :]

        segObj = segObj.astype(np.float32 )

        return segObj, segEnv, segArea


    def loadHdr(self, imName):
        if not(osp.isfile(imName ) ):
            print(imName )
            return None
        im = cv2.imread(imName, -1)
        if im is None:
            print(imName )
            return None
        im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :]
        im = np.ascontiguousarray(im )
        return im

    def scaleHdr(self, hdr, seg):
        # print(seg)
        # sdf
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.phase.upper() == 'TRAIN':
            scale = (0.9 - 0.1 * np.random.random() )  \
                    / np.clip(intensityArr[int(0.9 * self.imWidth * self.imHeight * 3) ], 0.1, None)
        elif self.phase.upper() == 'TEST':
            scale = (0.9 - 0.05)  \
                    / np.clip(intensityArr[int(0.9 * self.imWidth * self.imHeight * 3) ], 0.1, None)
        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale
    def process(self,img):
        img=img**(1.0/2.2)
        img*=255
        img=img.astype(np.uint8)
        img=img.transpose(1,2,0)
        return img[...,::-1]

import os
os.makedirs('TRAIN',exist_ok=True)
os.makedirs('TEST',exist_ok=True)

for stage in ['TRAIN','TEST']:
    loader=open40loader(phase=stage)
    imList,imDlList,imDmList=loader.get_imlist()
    labelList=[x.replace("im","imsemLabel").replace('hdr','npy') for x in imList]
    matList=[x.replace("im","immatPart").replace('hdr','dat') for x in imList]
    if not osp.isfile('nyuMap.npy'):
        nyuMap = np.zeros(46, dtype=np.uint16 )
        nyuMap[43 ] = 1
        nyuMap[44 ] = 2
        nyuMap[28 ] = 3
        nyuMap[42 ] = 3
        nyuMap[18 ] = 4
        nyuMap[21 ] = 5
        nyuMap[11 ] = 6
        nyuMap[4  ] = 7
        nyuMap[41 ] = 8
        nyuMap[31 ] = 9
        nyuMap[10 ] = 10
        nyuMap[7  ] = 12
        nyuMap[5  ] = 14
        nyuMap[1  ] = 16
        nyuMap[16 ] = 18
        nyuMap[45 ] = 22
        nyuMap[24 ] = 30
        nyuMap[32 ] = 35
        nyuMap[25 ] = 36
        nyuMap[37 ] = 37
        nyuMap[8  ] = 38
        nyuMap[26 ] = 38
        nyuMap[3  ] = 39
        nyuMap[6  ] = 39
        nyuMap[13 ] = 39
        nyuMap[14 ] = 39
        nyuMap[40 ] = 39
        nyuMap[9  ] = 40
        nyuMap[12 ] = 40
        nyuMap[15 ] = 40
        nyuMap[17 ] = 40
        nyuMap[19 ] = 40
        nyuMap[20 ] = 40
        nyuMap[22 ] = 40
        nyuMap[23 ] = 40
        nyuMap[27 ] = 40
        nyuMap[30 ] = 40
        nyuMap[33 ] = 40
        nyuMap[34 ] = 40
        nyuMap[35 ] = 40
        nyuMap[36 ] = 40
        nyuMap[38 ] = 40
        nyuMap[39 ] = 40
        nyuMap[2 ] = 0
        nyuMap[29 ] = 0   
        np.save('nyuMap.npy', nyuMap )
    else:
        nyuMap = np.load('nyuMap.npy' )
    final_result=[]
    bigger_index=-1
    for labelName, matName in zip(labelList,matList):
        bigger_index+=1
        clses, masks = fromSegToInsSeg(labelName, matName, nyuMap ) 
        labelNum = masks.shape[2] 
        objs_list=[]
        for ind  in range(0, labelNum ):
            mask = masks[:, :, ind] 
            ratio=1
            mask[0]=np.zeros(mask[0].shape)
            mask[-1]=np.zeros(mask[-1].shape)
            mask[:,0]=np.zeros(mask[:,0].shape)
            mask[:,-1]=np.zeros(mask[:,-1].shape)
            imHeight = 480
            imWidth = 640
            mask = cv2.resize(mask, (imWidth, imHeight),interpolation = cv2.INTER_AREA )
            contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            x1,y1=contours[0].min(axis=1).min(axis=0)
            x2,y2=contours[0].max(axis=1).max(axis=0)


            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                if len(contour) > 4:
                    segmentation.append(contour)
            if(len(segmentation)==0):
                continue
            # filter out the bike and flowerpot
            if(clses[ind]==0):
                continue
            obj = {
            "bbox": [x1,y1,x2,y2],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": segmentation,
            "category_id":clses[ind][0]-1,
            "iscrowd":0
            }
            objs_list.append(obj)
        # load imgs
        segObj, segEnv, segArea = loader.loadSeg(loader.segList[bigger_index ] )
        segObjDl, segEnvDl, segAreaDl = loader.loadSeg(loader.segDlList[bigger_index ]  )

        im_dir=loader.imList[bigger_index ] 
        im = loader.loadHdr(im_dir)
        if(im is None):
            continue
        im, scale = loader.scaleHdr(im, segObj )
        im=loader.process(im)
        imDl_dir=loader.imDlList[bigger_index ]
        imDl = loader.loadHdr(imDl_dir)
        if(imDl is None):
            continue
        imDl, scaleDl = loader.scaleHdr(imDl, segObjDl )
        imDl=loader.process(imDl)
        imDm_dir=loader.imDmList[bigger_index] 
        imDm = loader.loadHdr(imDm_dir)
        if(imDm is None):
            continue
        imDm, scaleDm = loader.scaleHdr(imDm, segObj )
        imDm=loader.process(imDm)
        # img=loader.loadHdr(imList[bigger_index])
        # imDl = loader.loadHdr(imDlList[bigger_index])
        # imDm = loader.loadHdr(imDmList[bigger_index])

        name1=prefix+"{}/{}.png".format(stage,bigger_index*3)
        name2=prefix+"{}/{}.png".format(stage,bigger_index*3+1)
        name3=prefix+"{}/{}.png".format(stage,bigger_index*3+2)

        cv2.imwrite(name1,im)
        cv2.imwrite(name2,imDl)
        cv2.imwrite(name3,imDm)

        more_objs={}
        more_objs["file_name"] = name1
        more_objs["image_id"] = bigger_index*3
        more_objs["height"] = imHeight
        more_objs["width"] = imWidth
        more_objs['annotations']=objs_list
        final_result.append(more_objs)

        more_objs["file_name"] = name2
        more_objs["image_id"] = bigger_index*3+1
        final_result.append(more_objs)

        more_objs["file_name"] = name3
        more_objs["image_id"] = bigger_index*3+2
        final_result.append(more_objs)


    # remove unqualified 
    def _make_array(t):
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        return np.asarray(t).astype("float64")
    count=0
    import copy
    dictt=copy.deepcopy(final_result)
    dictt2=copy.deepcopy(final_result)
    print('Before: ')
    print(len(dictt2))
    for item in dictt:
        for anno in item['annotations']:
            seg=anno['segmentation']
            polygons_per_instance = [_make_array(s) for s in seg]
            for polygon in polygons_per_instance:
                if(polygon is None or len(polygon)<4):
                    count+=1
                    try:
                        polygons_per_instance.remove(polygon)
                       # dictt2.remove(item)
                    except:
                        break
    print('After')
    print(len(dictt2))
    print('Removed :')
    print(count)
    final_result=dictt2
    torch.save(final_result,'{}/file.pth'.format(stage))
    print('having '+str(len(final_result))+' image in total')



