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
import os.path as osp
from tqdm import tqdm

######### remember to change checkpoint, True/ False and iteration number ##################
class BatchLoader(Dataset):
    def __init__(self, setup, dataRoot = '/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/DatasetCreation',
            imHeight = 480, imWidth = 640,
            phase='TRAIN', rseed = None,
            isLight = True,
            isLightBox = True,
            envHeight = 8, envWidth = 16, detect_4=False):
        #  whether reload the data
        self.detect_4=detect_4

        self.save_name='/file.pth'
        self.setup=setup
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()
        if self.phase=="TEST":
            # dataRoot='/eccv20dataset/DatasetNew_test/'
            dataRoot = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_test'
        self.dataRoot=dataRoot
        self.isLight = isLight
        self.isLightBox = isLightBox

        self.envWidth = envWidth
        self.envHeight = envHeight
        self.outdirr='/data/OR-detectron'
        # Load shapes
        shapeList = []
        dirs = ['main_xml', 'main_xml1']
        if phase.upper() == 'TRAIN':
            sceneFile = osp.join(dataRoot, 'train.txt')
            #self.outdir=dataRoot+'/train_data/'
            self.outdir=self.outdirr+'/train_data'
        elif phase.upper() == 'TEST':
            sceneFile = osp.join(dataRoot, 'test.txt')
            #self.outdir=dataRoot+'/test_data'
            self.outdir=self.outdirr+'/test_data'
        else:
            print('Wrong: unrecognizable phase')
            assert(False )
        if(self.detect_4):
            self.outdir+="_4class"
        if self.setup==True:
            with open(sceneFile, 'r') as fIn:
                sceneList = fIn.readlines()
            sceneList = [x.strip() for x in sceneList ]
            print(sceneFile)
            for d in dirs:
                shapeList = shapeList + [osp.join(dataRoot, d, x) for x in sceneList ]
            shapeList = sorted(shapeList )
            # all the sub-directory
            print('Shape Num: %d' % len(shapeList ) )

            # imList have len 36000, and imDlList and imDmList all have 36000 image, thus 11k images in total
            self.imList = []
            for num, shape in enumerate(shapeList):
                imNames = sorted(glob.glob(osp.join(shape, 'im_*.hdr') ) )
                self.imList = self.imList + imNames
            # self.imList=self.imList[:100]
            self.imDlList = [x.replace('main_', 'mainDiffLight_') for x in self.imList ]
            self.imDmList = [x.replace('main_', 'mainDiffMat_') for x in self.imList ]
            # all the hdr file name
            self.segList = [x.replace('im_', 'immask_').replace('hdr', 'png') for x in self.imList ]
            self.segDlList = [x.replace('main_', 'mainDiffLight_') for x in self.segList ]
            
            self.lightMaskList = []
            self.lightBoxList = []
            self.lightSrcList = []
            self.lightSrcDlList = []
            for x in self.imList:
                lightDir = x.replace('im_', 'light_').replace('.hdr', '')
                lightMaskNames = glob.glob(osp.join(lightDir, 'mask*.png') )
                lightBoxNames = [x.replace('mask', 'box').replace('.png', '.dat')
                        for x in lightMaskNames ]
                # print(lightMaskNames) # ['/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_test/main_xml1/scene0458_00/light_17/box1.dat', '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_test/main_xml1/scene0458_00/light_17/box0.dat']
                # print(lightBoxNames) # ['/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_test/main_xml1/scene0458_00/light_18/mask1.png', '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_test/main_xml1/scene0458_00/light_18/mask0.png']

                self.lightMaskList.append(lightMaskNames )
                self.lightBoxList.append(lightBoxNames )

            # Permute the image list
            self.count = len(self.imList )
            self.perm = list(range(self.count ) )
        # if rseed is not None:
            # random.seed(0)
        # random.shuffle(self.perm )

    def __len__(self):
        return len(self.perm )

    def run(self):
        # outfiledir=self.outdirr
        if(self.setup==True):
            # map1={}
            if (os.path.isdir(self.outdir)):
                os.system("rm -rf {}".format(self.outdir))
            os.mkdir(self.outdir)
            final_result=[]
            # file_=open(self.outdir+'/map.txt','w')
            for ind in tqdm(range(len(self.perm))):
                ############### checkpoint ######################
                if(ind>0 and ind%1000==0):
                    print('{} out of {} finished!'.format(ind,len(self.perm)))
                    torch.save(final_result,self.outdir+self.save_name)

                segObj, segEnv, segArea = self.loadSeg(self.segList[self.perm[ind ] ] )
                segObjDl, segEnvDl, segAreaDl = self.loadSeg(self.segDlList[self.perm[ind] ]  )

                segAll = np.clip(segObj + segArea, 0, 1 )

                lightMaskNames = self.lightMaskList[self.perm[ind ] ]
                lightBoxNames = self.lightBoxList[self.perm[ind ] ]

                # light prediction for invisible light source
                lightBoxs = []
                lightBoxsDl = []
                lightBoxsDist = []
                lightBoxsDlDist = []

                # raw parameters for light source
                objs_list=[]
                # for each masks label corresponding to each .hdr image
                for n in range(0, len(lightMaskNames ) ):
                    record={}
                    maskName = lightMaskNames[n ]
                    boxName = lightBoxNames[n ]
                    mask = cv2.imread(maskName )[:, :, 0]
                    mask = cv2.resize(mask, (self.imWidth, self.imHeight),
                            interpolation = cv2.INTER_AREA )
                    mask = mask.astype(np.float32 ) / 255.0
                    with open(boxName, 'rb') as fIn:
                        boxPara = pickle.load(fIn )
                    # if there's no box2D label but box3D, then the light source is invisible
                    if(boxPara['box2D']['x1']==None):
                        continue
                    else:
                        box=boxPara['box2D']

                        lightt_name=boxName.replace('box','light')
                        intensity=sum(pickle.load(open(lightt_name,'rb'))['intensity'])
                        is_window=boxPara['isWindow']==True
                        is_close=intensity==0
                        # for two class detection, don't need to detect close lamp/window
                        if(is_close==True and self.detect_4==False):
                            continue

                        # since the image is resize, we need to resize bbox as well
                        ratio=1
                        x1,x2,y1,y2=box['x1']*ratio,box['x2']*ratio,box['y1']*ratio,box['y2']*ratio

                        # turn binary mask into polygon (RLE format)
                        contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                        segmentation = []
                        # in here, contours is a list of contour of small region, sometime it will have some extra contour of length 1 or 2
                        # thus we only keep those contour of more than 3 dots
                        for contour in contours:
                            contour = contour.flatten().tolist()
                            '''
                            if len(contour)<6:
                                im=self.process(im)
                                cv2.imwrite("{}.png".format(ind),im )
                                import pdb;pdb.set_trace()
                            '''
                            if len(contour) > 6:
                                segmentation.append(contour)
                        # contour=measure.find_contours(mask, 0.5)
                        if(len(segmentation)==0):
                            continue
                        obj = {
                            "bbox": [x1,y1,x2,y2],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": segmentation,
                            "iscrowd":0
                            }
                        # window on :0 off: 1, lamp on:2, off :3
                        if self.detect_4:
                            idd=-1
                            if is_window:
                                if is_close:
                                    idd=1
                                else:
                                    idd=0
                            else:
                                if is_close:
                                    idd=3
                                else:
                                    idd=2
                            obj["category_id"]= idd
                        else:
                            obj["category_id"]= 0 if boxPara['isWindow']==True else 1

                        objs_list.append(obj)
                        ################# Data Augumentation #################################
                
                im_dir=self.imList[self.perm[ind ] ]

                im = self.loadHdr(im_dir)
                if(im is None):
                    continue
                im, scale = self.scaleHdr(im, segObj )

                imDl_dir=self.imDlList[self.perm[ind ] ]
                imDl = self.loadHdr(imDl_dir)
                if(imDl is None):
                    continue
                imDl, scaleDl = self.scaleHdr(imDl, segObjDl )

                imDm_dir=self.imDmList[self.perm[ind] ]
                imDm = self.loadHdr(imDm_dir)
                if(imDm is None):
                    continue
                imDm, scaleDm = self.scaleHdr(imDm, segObj )

                # write the process .hdr image to png for later training
                # id should be 0 to 11k
                filename=osp.join(self.outdir,im_dir.replace("/","-").replace("hdr","png"))
                for i in range(3):
                    if(i==0):
                        id=3*ind
                        im=self.process(im)
                        filename=osp.join(self.outdir,im_dir.replace("/","-").replace("hdr","png"))
                        cv2.imwrite(filename, im)
                    elif(i==1):
                        id=3*ind+1
                        imDl=self.process(imDl)
                        filename=osp.join(self.outdir,imDl_dir.replace("/","-").replace("hdr","png"))
                        cv2.imwrite(filename, imDl)
                    else:
                        id=3*ind+2
                        imDm=self.process(imDm)
                        filename=osp.join(self.outdir,imDm_dir.replace("/","-").replace("hdr","png"))
                        cv2.imwrite(filename, imDm)
                    # if there's no visible light source for this .hdr image
                    if(len(objs_list)!=0):
                        more_objs={}
                        more_objs["file_name"] = filename
                        more_objs["image_id"] = id
                        more_objs["height"] = self.imHeight
                        more_objs["width"] = self.imWidth
                        more_objs['annotations']=objs_list
                        final_result.append(more_objs)

                    else:
                        more_objs={}
                        more_objs["file_name"] = filename
                        more_objs["image_id"] = id
                        more_objs["height"] = self.imHeight
                        more_objs["width"] = self.imWidth
                        more_objs['annotations']=[]
                        final_result.append(more_objs)
            self.dict=final_result
            torch.save(self.dict,self.outdir+self.save_name)
            print('having '+str(len(self.dict))+' image in total')
        else:
            self.dict=torch.load(self.outdir+self.save_name)
            
    def __getitem__(self, k):
        return self.dict[k]



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
