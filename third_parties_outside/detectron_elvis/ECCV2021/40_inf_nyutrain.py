# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# start of my code
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
########################################   trainning  ######################################
import datetime
import shutil
import torch
######  register 
#outfiledir='/siggraphasia20dataset/code/Routine/elvis/outfileelvis/'
outfiledir='/eccv20dataset/elvis/outfileelvis/'
outrecord=outfiledir+'record.txt'
record=open(outrecord,'a')

test_dict=torch.load("/eccv20dataset/elvis/NYU/test_40/file.pth")

for d in ["test"]:
    DatasetCatalog.register("light_" + d, lambda d=d:test_dict)
    #MetadataCatalog.get("light_" + d).set(thing_classes=["window_on","window_off",'lamp_on',"lamp_off"])
    MetadataCatalog.get("light_" + d).set(thing_classes=[str(kk) for kk in range(1,41)])
light_metadata_test = MetadataCatalog.get("light_test")

print('Final register Testing data!')

with open(outrecord,'a') as record:
    record.write('finish test resigter, {}  \n'.format(str(datetime.datetime.now())))


dictt=torch.load("/eccv20dataset/elvis/NYU/train_40/file.pth")

for d in ["train"]:
    DatasetCatalog.register("light_" + d, lambda d=d:dictt)
    #MetadataCatalog.get("light_" + d).set(thing_classes=["window_on","window_off",'lamp_on',"lamp_off"])
    MetadataCatalog.get("light_" + d).set(thing_classes=[str(kk) for kk in range(1,41)])
light_metadata = MetadataCatalog.get("light_train")

print('Final register Training data!')

with open(outrecord,'a') as record:
    record.write('finish train resigter, {} \n'.format(str(datetime.datetime.now())))



from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import torch
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("light_train",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER =40000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 40
# cfg.MODEL.TENSOR_MASK.MASK_LOSS_WEIGHT = 3
cfg.SOLVER.WEIGHT_DECAY: 0.0001
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST: 0.7
cfg.SOLVER.MOMENTUM: 0.9
cfg.SOLVER.LR_SCHEDULER_NAME="WarmupCosineLR"
cfg.OUTPUT_DIR=outfiledir+'out_only_nyu_37_class'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
'''
try:
    trainer.train()
except:
    pass
'''

########################################   Evaluation  ######################################
# eval_dir='/siggraphasia20dataset/code/Routine/DatasetCreation/eval_sample'
# if os.path.isdir(eval_dir)==True:
#     shutil.rmtree(eval_dir)
# os.mkdir(eval_dir)
from detectron2.utils.visualizer import ColorMode
print(cfg.OUTPUT_DIR)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   
cfg.DATASETS.TEST = ("light_test", )
predictor = DefaultPredictor(cfg)
# a file is the map between origin image and agent image
# (/siggraphasia20dataset/code/Routine/DatasetCreation/train(test)_data/map.txt)
# b file is the map between agent image and prediction mask
# /siggraphasia20dataset/code/Routine/elvis/ECCV2021/map2_predictionTRAIN(TEST).txt

def final_mapping(a_file,b_file,stage):
    import torch
    list_origin2agent={}
    list_agent2mask={}
    list_origin2mask={}
    origin2agent=open(a_file,'r')
    while True:
        # read all the line
        try:
            line=origin2agent.readline().split()
            list_origin2agent[line[1]]=line[0]
        except:
            break
    agent2mask=open(b_file,'r')
    while True:
        # read all the line
        try:
            line=agent2mask.readline().split()
            list_agent2mask[line[1]]=line[0]
        except:
            break
    for key, value in list_origin2agent.items():
        try:
            list_origin2mask[key]=list_agent2mask[value]
        except:
            continue
    torch.save(list_origin2mask,outfiledir+'image2map{}.pth'.format(stage))
def py_cpu_nms(imgs, scores,thresh):
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
            print('error')
            return sorted(keep)
    return sorted(keep)
with open(outrecord,'a') as record:
    record.write('finish training, {} \n'.format(str(datetime.datetime.now())))
#####################3
#map between output mask and input mask
for stage in['TEST','TRAIN']:
    if(stage=='TRAIN'):
        dataset_dicts=dictt
        data_folder='train_data'
    else:
        dataset_dicts=test_dict
        data_folder='test_data'
    print('starting evaluating/ generating mask')
    pred_path=outfiledir+'output_mask{}'.format(stage)
    if os.path.isdir(pred_path)==True:
        shutil.rmtree(pred_path)    
    os.mkdir(pred_path)
    eval_dir=outfiledir+'eval_sample{}'.format(stage)
    if os.path.isdir(eval_dir)==True:
        shutil.rmtree(eval_dir)    
    os.mkdir(eval_dir)
    file_=open(outfiledir+'map2_prediction{}.txt'.format(stage), 'w')
    # map1=torch.load('./map1_{}.torch'.format(stage))
    count=0
    for indd,d in enumerate(dataset_dicts):
        try:
            im = cv2.imread(d["file_name"])
        except:
            print(d)
            continue
        outputs = predictor(im)
        dictt=outputs['instances']._fields
        # put all the predction of the current image into a list, them apply NMS
        scores=dictt['scores'].cpu().numpy()
        mask_pre_list=(dictt['pred_masks'].cpu().numpy()*255).astype('uint8')[scores>0.8]
        if(mask_pre_list.shape[0]==0):
            continue
        mask_pre_list=mask_pre_list[py_cpu_nms(mask_pre_list, scores,0.8)]
        # mask_list=map1[d["file_name"]]
        ## visualize the prediction mask
        v = Visualizer(im[:, :, ::-1], metadata=light_metadata_test, scale=0.4)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(eval_dir+'/{}eva.png'.format(count),v.get_image()[:, :, ::-1])
        import numpy as np
        import torch
        
        mask_concat=np.stack(mask_pre_list)
        torch.save(mask_concat,pred_path+'/{}.pth'.format(count))
        with open(outfiledir+'map2_prediction{}.txt'.format(stage), 'a') as out:
             out.write(pred_path+'/{}.pth'.format(count)+ ' ' + d["file_name"]+'\n')
        count+=1
    final_mapping(outfiledir+'{}/map.txt'.format(data_folder),outfiledir+'map2_prediction{}.txt'.format(stage),stage)
with open(outrecord,'a') as record:
    record.write('finish evaluating, {} \n'.format(str(datetime.datetime.now())))

print('finish evalting generating images')
'''
## for each output mask in prediction
    for mask_pre in mask_pre_list:
        ## find the gt mask that have the greatest iou with the prediction  mask
        mask_iou=np.array([np.sum(cv2.bitwise_and(mask_pre,cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE))/255) for mask_name in mask_list])
        ## get the name of the gt mask
        max_name=mask_list[mask_iou.argmax()]
        #output map of mask_pred and mask_gt
        output_name=pred_path+'/{}mask.png'.format(count)
        with open('./map2_prediction{}.txt'.format(stage), 'a') as out:
            out.write(output_name+" "+max_name+"\n")
        cv2.imwrite(output_name,mask_pre)



from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import shutil
for stage in ['test','train']:
    eval_dir="../outfileelvis/output_{}/".format(stage)
    if os.path.isdir(eval_dir)==True:
         shutil.rmtree(eval_dir)
    evaluator = COCOEvaluator("light_{}".format(stage), ("bbox", "segm"), False, output_dir=eval_dir)
    val_loader = build_detection_test_loader(cfg, "light_{}".format(stage))
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
'''
