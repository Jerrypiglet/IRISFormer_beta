import torch
import numpy as np
import cv2
# when torch is older, it would not load properly
try:
    assert torch.__version__=='1.6.0'
except:
    os. system("pip3 install torch==1.6.0")
    os. system("pip3 install -U opencv-python==4.2.0.34")





for stage in['TEST']:
    # map from hdr name to number name
    map_dict={}
    origin2agent=open('../outfileelvis/{}_data/map.txt'.format(stage.lower()),'r')
    while True:
        # read all the line
        try:
            line=origin2agent.readline().split()
            map_dict[line[0]]=line[1]
        except:
            break
    accuracy=[]
    # mapping from input to ground truth mask
    map1_name="../outfileelvis/map1_{}.torch".format(stage)
    # mapping from input to predicted mask
    map2_name="../outfileelvis/image2map{}.pth".format(stage)
    map1=torch.load(map1_name)
    map2=torch.load(map2_name)
    keys=map1.keys()
    count=0
    for img in keys:
        count+=1
        if(count%100==0):
            print(count)
        try:
            value=map1[img]
            # get the added mask for ground truth mask1
            if len(value)==0:
                continue
            if len(value)==1:
                mask1=cv2.imread(value[0],0)
            else:
                mask1=cv2.imread(value[0],0)
                for gt in value:
                    mask1=cv2.bitwise_or(mask1, cv2.imread(gt,0))
            # get the added mask for predict mask2 
            img=map_dict[img]
            mask2=torch.load(map2[img].replace('/siggraphasia20dataset/code/Routine/elvis/','../'))
            mask2=np.sum(mask2,0)
            # print(mask1.shape)
            # print(mask2.shape)
            # print(mask1.dtype)
            # print(mask2.dtype)
            mask1=np.array(mask1,dtype=np.int64)
            mask2=np.array(mask2,dtype=np.int64)
            accu=np.sum(cv2.bitwise_and(mask1,mask2))/np.sum(mask1)
            if float(type(accu))==float and np.isnan(accu)==False:
                accuracy.append(accu)
        except:
            continue
    result=np.mean(accuracy)
    print(accuracy)
    print("The segmentation accuracy for {} is {}".format(stage,result))



