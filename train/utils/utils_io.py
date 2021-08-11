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
        if if_resize:
            if dtype == np.float32:
                depth = cv2.resize(depth, (resize_HW[1], resize_HW[0]), interpolation=cv2.INTER_AREA )
            elif dtype == np.int32:
                depth = cv2.resize(depth.astype(np.float32), (resize_HW[1], resize_HW[0]), interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.int32)

        depth = np.squeeze(depth)

    return depth
