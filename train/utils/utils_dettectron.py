import numpy as np
import cv2

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
