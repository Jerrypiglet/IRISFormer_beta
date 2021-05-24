import os
import json
import datetime
import argparse
import torch
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader,DatasetCatalog, MetadataCatalog
import os.path as osp
from detectron2.engine import DefaultTrainer
from elvisloader import *


parser = argparse.ArgumentParser()
parser.add_argument('--folder', default="../outfileelvis/test_data/", type=str, nargs='?',
                    help='inference folder path')
parser.add_argument('--thresh', default=0.75, type=float, nargs='?',
                    help='threshold of filtering confidence score in inference')
parser.add_argument('--nms_thresh', default=0.6, type=float, nargs='?',
                    help='threshold of filtering out mask in NMS')
parser.add_argument('--detect_4', default=0, type=int, nargs='?',
                    help='whether to detech 4 classes(light on/off + lamp/window), else, its lamp window light on ')

def py_cpu_nms(imgs, scores,thresh):
    if(imgs.shape[0]==1):
        return [0]
    order = scores.argsort()[::-1]  
    keep = []
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        mask_pre=imgs[i]
        try:
            ovr=np.array([np.sum(cv2.bitwise_and(mask_pre,image)) /np.sum(image) for image in imgs[order[1:]]])
            inds = np.where(ovr <= thresh)[0] 
            order = order[inds + 1] 
        except:
            # import pdb;pdb.set_trace()
            # often the size of imgs and scores not match, fixed
            print('NMS error in inference!')
            return sorted(keep)
    return sorted(keep)


# preidct all 11w image in data folder, not only the trained/tested image
def inference(img_folder_name):
    print(cfg.OUTPUT_DIR)
    cfg.MODEL.WEIGHTS = osp.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   
    predictor = DefaultPredictor(cfg)
    img_folder=os.listdir(img_folder_name)
    #visulation of the result map 
    for image_name in img_folder:
        im = cv2.imread(os.path.join(img_folder_name,image_name))
        outputs = predictor(im)

        output_dict=outputs['instances']._fields
        # put all the predction of the current image into a list, them apply NMS
        scores=output_dict['scores'].cpu().numpy()
        mask_pre_list=(output_dict['pred_masks'].cpu().numpy()*255).astype('uint8')[scores>args.thresh]
        scores=scores[scores>args.thresh]
        if(mask_pre_list.shape[0]==0):
            print('empty inference')
            mask=np.zeros(im.shape)
            #return [mask]
        else:
            # the list of masks, [mask1,mask2,mask3....]
            mask_pre_list_filtered=mask_pre_list[py_cpu_nms(mask_pre_list, scores,args.nms_thresh)]
            print('mask shape : {} '.format(len(mask_pre_list_filtered)))




if __name__ == '__main__':
    global outfiledir, cfg
    cfg = get_cfg()

    outfiledir='/eccv20dataset/elvis/outfileelvis/'
    args = parser.parse_args()
    print(args)




    tester=BatchLoader(setup=0,phase='TEST',detect_4=args.detect_4)
    tester.run()
    test_dict=tester.dict

    for d in ["test"]:
        DatasetCatalog.register("light_" + d, lambda d=d:test_dict)
        if(args.detect_4):
            MetadataCatalog.get("light_" + d).set(thing_classes=["window_on","window_off",'lamp_on',"lamp_off"])
        else:
            MetadataCatalog.get("light_" + d).set(thing_classes=["window",'lamp'])
    light_metadata_test = MetadataCatalog.get("light_test")

    print('Final register Testing data!')


    trainerr=BatchLoader(setup=0,phase='TRAIN',detect_4=args.detect_4)
    trainerr.run()

    dictt=trainerr.dict

    for d in ["train"]:
        DatasetCatalog.register("light_" + d, lambda d=d:dictt)
        if(args.detect_4):
            MetadataCatalog.get("light_" + d).set(thing_classes=["window_on","window_off",'lamp_on',"lamp_off"])
        else:
            MetadataCatalog.get("light_" + d).set(thing_classes=["window",'lamp'])
    light_metadata = MetadataCatalog.get("light_train")

    print('Final register Training data!')


    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("light_train",)
    cfg.DATASETS.TEST = ("light_test", )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    if(args.detect_4):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.SOLVER.WEIGHT_DECAY: 0.0001
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST: 0.7
    cfg.SOLVER.MOMENTUM: 0.9
    cfg.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
    cfg.OUTPUT_DIR=outfiledir+'model_path/'
    # False to include all empty image
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    if(args.detect_4):
        cfg.OUTPUT_DIR=outfiledir+'model_path_4class/'
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    image_name=args.folder
    result=inference(image_name)
