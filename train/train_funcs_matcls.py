import numpy as np
import torch
from torch.autograd import Variable
# import models
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import statistics
import time
import torchvision.utils as vutils
from utils.utils_vis import vis_index_map, reindex_output_map
from utils.utils_training import reduce_loss_dict, time_meters_to_string
from utils.utils_misc import *
from PIL import Image

ceLoss = nn.CrossEntropyLoss()
ceLoss_sup = nn.CrossEntropyLoss(ignore_index=0) # ignore mat cls sup == 0 for unlabelled and homogeneous materials

def get_labels_dict_matcls(data_batch, opt):
    input_dict = {}

    im_cpu = (data_batch['im_trainval'].permute(0, 3, 1, 2) )
    input_dict['imBatch'] = Variable(im_cpu ).cuda(non_blocking=True)

    input_dict['mat_mask_batch'] = Variable(data_batch['matMask']).cuda(non_blocking=True)
    input_dict['mat_label_batch'] = Variable(data_batch['matLabel']).cuda(non_blocking=True)

    if opt.cfg.MODEL_MATCLS.if_est_sup:
        input_dict['mat_label_sup_batch'] = Variable(data_batch['matLabelSup']).cuda(non_blocking=True)
    return input_dict

def postprocess_matcls(input_dict, output_dict, loss_dict, opt, time_meters, if_vis=False):
    matcls_output, matcls_argmax = output_dict['matcls_output'], output_dict['matcls_argmax']

    # ======= Calculate loss
    classErr = ceLoss(matcls_output, input_dict['mat_label_batch'].long())
    loss_dict['loss_matcls-ALL'] = classErr
    loss_dict['loss_matcls-cls'] = classErr
    # loss_dict['loss_matseg-push'] = torch.mean(torch.stack(loss_push_list))
    # loss_dict['loss_matseg-binary'] = torch.mean(torch.stack(loss_binary_list))

    if opt.cfg.MODEL_MATCLS.if_est_sup:
        matcls_sup_output, matcls_sup_argmax = output_dict['matcls_sup_output'], output_dict['matcls_sup_argmax']
        classErr_sup = ceLoss_sup(matcls_sup_output, input_dict['mat_label_sup_batch'].long())
        loss_dict['loss_matcls-ALL'] += classErr_sup * opt.cfg.MODEL_MATCLS.loss_sup_weight
        loss_dict['loss_matcls-supcls'] = classErr_sup

    return output_dict, loss_dict

def getG1IdDict(matIdG1File):
    #matG1File = osp.join(dataRoot, 'matIdGlobal1.txt')
    matG1Dict = {}
    with open(matIdG1File, 'r') as f:
        for line in f.readlines():
            if 'Material__' not in line:
                continue
            matName, mId = line.strip().split(' ')
            matG1Dict[int(mId)] = matName
    return matG1Dict

def getRescaledMatFromID(matIds, matScales, oriMatRoot, matG1IdDict, res=128):
    # Apply scaled to specific materials
    svbrdfList = os.listdir(oriMatRoot)
    albedos = []
    normals = []
    roughs = []
    mats = []
    for i, matId in enumerate(matIds):
        matName = matG1IdDict[matId+1]
        rgbScale = matScales[i, :3]
        roughScale = matScales[i, 3]
        if matName in svbrdfList:  # is svbrdf
            albedoFile = os.path.join(oriMatRoot, matName,
                                      'tiled', 'diffuse_tiled.png')
            albedo = Image.open(albedoFile).convert('RGB')
            albedo = albedo.resize((res, res), Image.ANTIALIAS)
            albedo = np.asarray(albedo) / 255.0
            albedo = np.clip(
                (albedo ** 2.2) * rgbScale[np.newaxis, np.newaxis, :], 0.0, 1.0) ** (1/2.2)
            # albedo = Image.fromarray(np.uint8(albedo * 255))
            normal = np.asarray(Image.open(
                albedoFile.replace('diffuse', 'normal') ).resize((res, res), Image.ANTIALIAS) ) / 255.0

            rough = Image.open(albedoFile.replace(
                'diffuse', 'rough')).convert('L').resize((res, res), Image.ANTIALIAS)
            rough = np.asarray(rough) / 255.0
            rough = np.clip(rough * roughScale, 0.0, 1.0)
            rough = np.tile(rough[:,:,np.newaxis], (1, 1, 3))
            # rough = Image.fromarray(np.uint8(rough * 255))
        else:  # is homogeneous brdf
            _, vals = matName.split('__')
            r, g, b, rough = vals.split('_')
            rgb = np.array([float(r), float(g), float(b)])
            rgb = np.clip(rgb * rgbScale, 0.0, 1.0) ** (1/2.2)
            albedo = np.tile(rgb, (res, res, 1))
            normal = np.tile(np.array([0.5, 0.5, 1]), [res, res, 1])
            rough = np.clip(float(rough) * roughScale, 0.0, 1.0)
            rough = np.tile(rough, (res, res, 3))
        # albedos.append(th.from_numpy(np.transpose(albedo, [2, 0, 1])))
        # normals.append(th.from_numpy(np.transpose(normal, [2, 0, 1])))
        # roughs.append(th.from_numpy(np.transpose(rough, [2, 0, 1])))
        # print(albedo.shape) # [256, 256, 3]
        mat = np.concatenate([albedo, normal, rough], axis=1) # [D, 3D, 3]
        mats.append(torch.from_numpy(np.transpose(mat, [2, 0, 1])))

    return mats

