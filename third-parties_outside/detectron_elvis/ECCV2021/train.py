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
parser.add_argument('--traindata', default=0, type=int, nargs='?',
                    help='whether to generate the training loader')
parser.add_argument('--testdata', default=0, type=int, nargs='?',
                    help='whether to generate the testing loader')
parser.add_argument('--train', default=0, type=int, nargs='?',
                    help='whether to run the training ')
parser.add_argument('--eval', default=0, type=int, nargs='?',
                    help='whether to run the evaluation ')
parser.add_argument('--infer', default=0, type=int, nargs='?',
                    help='whether to run the inference ')
parser.add_argument('--iter', default=10000, type=int, nargs='?',
                    help='interation of training')
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

def train(trainer):
    try:
        trainer.train()
    except:
        pass
    return trainer

# preidct all 11w image in data folder, not only the trained/tested image
def inference(dictt, test_dict):
    print(cfg.OUTPUT_DIR)
    cfg.MODEL.WEIGHTS = osp.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   
    predictor = DefaultPredictor(cfg)
    #visulation of the result map 
    for stage in['TEST','TRAIN']:
        if(stage=='TRAIN'):
            # dataset_dicts=dictt
            data_folder='train_data'
        else:
            # dataset_dicts=test_dict
            data_folder='test_data'
        if(args.detect_4):
            data_folder+="_4class"

        print("*******************************************************************************************************************************************")
        print("*************************************************** Starting inference/ generating pred mask& viz *****************************************")
        print("*******************************************************************************************************************************************")
        print("Stage " +stage)
        pred=osp.join(outfiledir,"pred_"+data_folder)
        os.system("rm -rf {}".format(pred))
        os.mkdir(pred)
        viz=osp.join(outfiledir,"viz_"+data_folder)
        os.system("rm -rf {}".format(viz))
        os.mkdir(viz)
        count=0
        data_folder=osp.join(outfiledir,data_folder)
        image_list=os.listdir(data_folder)
        for cur_ind, image_name in enumerate(image_list):
            if(cur_ind%100==0):
                print("{} out of {} finished".format(cur_ind,len(image_list)))
        # for indd,d in enumerate(dataset_dicts):
            if ".pth" in image_name:
                continue
            im = cv2.imread(osp.join(data_folder, image_name))
            filename=image_name.replace("-","/")
            filename="/".join(filename.split("/")[-3:])
            # filesubpath= main_xml/scene0006_00/im_11
            filesubpath=osp.splitext(filename)[0]
            outputs = predictor(im)

            output_dict=outputs['instances']._fields
            # put all the predction of the current image into a list, them apply NMS
            scores=output_dict['scores'].cpu().numpy()
            mask_pre_list=(output_dict['pred_masks'].cpu().numpy()*255).astype('uint8')[scores>args.thresh]
            scores=scores[scores>args.thresh]
            os.makedirs(osp.join(pred,filesubpath), exist_ok=True)
            os.makedirs(osp.join(viz,filesubpath), exist_ok=True)
            if(mask_pre_list.shape[0]==0):
                mask=np.zeros(im.shape)
                cv2.imwrite(osp.join(pred,filesubpath,'empty_mask.png'), mask)
                cv2.imwrite(osp.join(viz,filesubpath,'empty_mask.png'), mask)
            else:
                mask_pre_list=mask_pre_list[py_cpu_nms(mask_pre_list, scores,args.nms_thresh)]
                # mask_list=map1[d["file_name"]]
                ## visualize the prediction mask
                v = Visualizer(im[:, :, ::-1], metadata=light_metadata_test, scale=1)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite(osp.join(viz,filesubpath,'viz.png'),v.get_image()[:, :, ::-1])
                for index, mask in enumerate(mask_pre_list):
                    cv2.imwrite(osp.join(pred,filesubpath,'mask{}.png'.format(index)), mask)



    # ## for each output mask in prediction/ find the mapping to the original label
    # for mask_pre in mask_pre_list:
    #     ## find the gt mask that have the greatest iou with the prediction  mask
    #     mask_iou=np.array([np.sum(cv2.bitwise_and(mask_pre,cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE))/255) for mask_name in mask_list])
    #     ## get the name of the gt mask
    #     max_name=mask_list[mask_iou.argmax()]
    #     #output map of mask_pred and mask_gt
    #     output_name=pred_path+'/{}mask.png'.format(count)
    #     with open('./map2_prediction{}.txt'.format(stage), 'a') as out:
    #         out.write(output_name+" "+max_name+"\n")
    #     cv2.imwrite(output_name,mask_pre)

def eval(trainer):
    print("*******************************************************************************************************************************************")
    print("******************************************* Starting evaluation for Bounding Box and Segmentation *****************************************")
    print("*******************************************************************************************************************************************")
    for stage in ['test','train']:
        print("Stage " +stage)
        eval_dir=outfiledir+"eval_{}/".format(stage)
        if osp.isdir(eval_dir)==True:
             os.system("rm -rf {}".format(eval_dir))
        evaluator = COCOEvaluator("light_{}".format(stage), ("bbox", "segm"), False, output_dir=eval_dir)
        val_loader = build_detection_test_loader(cfg, "light_{}".format(stage))
        print(inference_on_dataset(trainer.model, val_loader, evaluator))



if __name__ == '__main__':
    global outfiledir, cfg
    cfg = get_cfg()

    outfiledir='/eccv20dataset/elvis/outfileelvis/'
    args = parser.parse_args()
    print(args)




    tester=BatchLoader(setup=args.testdata,phase='TEST',detect_4=args.detect_4)
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


    trainerr=BatchLoader(setup=args.traindata,phase='TRAIN',detect_4=args.detect_4)
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
    cfg.SOLVER.MAX_ITER = args.iter    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    if(args.detect_4):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    # cfg.MODEL.TENSOR_MASK.MASK_LOSS_WEIGHT = 3
    cfg.SOLVER.WEIGHT_DECAY: 0.0001
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST: 0.7
    cfg.SOLVER.MOMENTUM: 0.9
    cfg.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
    cfg.OUTPUT_DIR=outfiledir+'model_path/'
    # False to include all empty image
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    if(args.detect_4):
        cfg.OUTPUT_DIR=outfiledir+'model_path_4class/'
    if args.train:
        os.system("rm -rf {}".format(cfg.OUTPUT_DIR))
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer=train(trainer)
    else:
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=True)
    if args.infer:
        inference(dictt,test_dict)
    if args.eval:
        eval(trainer)
