import h5py
import numpy as np
import struct
import os.path as osp
import cv2

def loadH5(imName ): 
    try:
        hf = h5py.File(imName, 'r')
        im = np.array(hf.get('data' ) )
        return im 
    except:
        return None

def loadBinary(imName, channels = 1, dtype=np.float32, resize_HW=[-1, -1]):
    assert dtype in [np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
    if_resize = resize_HW!=[-1, -1]

    if not(osp.isfile(imName ) ):
        assert(False ), '%s doesnt exist!'%imName
    with open(imName, 'rb') as fIn:
        hBuffer = fIn.read(4)
        height = struct.unpack('i', hBuffer)[0]
        wBuffer = fIn.read(4)
        width = struct.unpack('i', wBuffer)[0]
        dBuffer = fIn.read(4 * channels * width * height )
        if dtype == np.float32:
            decode_char = 'f'
        elif dtype == np.int32:
            decode_char = 'i'
        depth = np.asarray(struct.unpack(decode_char * channels * height * width, dBuffer), dtype=dtype)
        depth = depth.reshape([height, width, channels] )

        # print(depth.shape)

        if if_resize:
            if dtype == np.float32:
                depth = cv2.resize(depth, (resize_HW[1], resize_HW[0]), interpolation=cv2.INTER_AREA )
            elif dtype == np.int32:
                depth = cv2.resize(depth.astype(np.float32), (resize_HW[1], resize_HW[0]), interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.int32)

        depth = np.squeeze(depth)
        # print('>>>>', depth.shape)

    return depth


from PIL import Image

def loadImage(imName=None, im=None, isGama = False, resize_HW=[-1, -1]):

    if im is None:
        if not(osp.isfile(imName ) ):
            assert(False), 'File does not exist: ' + imName 
        im = Image.open(imName)
        if_resize = resize_HW!=[-1, -1]
        if if_resize:
            im = im.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
        im = np.asarray(im, dtype=np.float32)
        
    if isGama:
        im = (im / 255.0) ** 2.2
        im = 2 * im - 1
    else:
        im = (im - 127.5) / 127.5
    if len(im.shape) == 2:
        im = im[:, np.newaxis]
    im = np.transpose(im, [2, 0, 1] )

    return im

