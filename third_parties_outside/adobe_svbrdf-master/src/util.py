import os
import sys
import glob
import math
import json
import random
import argparse
import numpy as np
import torch as th
import torchvision
from PIL import Image
# import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

eps = 1e-6


def gyCreateFolder(dir):
    if not os.path.exists(dir):
        print("\ncreate directory: ", dir)
        os.makedirs(dir)


def gyListNames(in_dir):
    dir_list = sorted(glob.glob(in_dir))
    fn_list = []
    for dir in dir_list:
        fn_list.append(os.path.split(dir)[1])
    return fn_list


def gyConcatPIL_h(im1, im2):
    if im1 is None:
        return im2
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def gyConcatPIL_v(im1, im2):
    if im1 is None:
        return im2
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def gyPIL2Array(im):
    return np.array(im).astype(np.float32)/255


def gyArray2PIL(im):
    return Image.fromarray((im*255).astype(np.uint8))


def gyApplyGamma(im, gamma):
    if gamma < 1:
        im = im.clip(min=eps)
    return im**gamma


def gyApplyGammaPIL(im, gamma):
    return gyArray2PIL(gyApplyGamma(gyPIL2Array(im), gamma))


def gyTensor2Array(im):
    return im.detach().cpu().numpy()


def gyCreateThumbnail(fnA, w=128, h=128):
    fnB = os.path.join(os.path.split(fnA)[0], 'jpg')
    gyCreateFolder(fnB)
    fnB = os.path.join(fnB, os.path.split(fnA)[1][:-3]+'jpg')
    os.system('convert ' + fnA + ' -resize %dx%d -quality 100 ' % (w, h) + fnB)


def loadTex(imPath):
    im = Image.open(imPath)
    return np.array(im).astype(np.float32)/255


def printMSG(msg, log):
    print(msg)
    log.write(msg+'\n')


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
        mat = np.concatenate([albedo, normal, rough], axis=0)
        mats.append(th.from_numpy(np.transpose(mat, [2, 0, 1])))

    return mats
