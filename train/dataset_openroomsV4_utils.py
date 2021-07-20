import numpy as np
import os.path as osp
import struct
import cv2

def loadBinary(imName, channels = 1, dtype=np.float32, if_resize=False, im_width=-1, im_height=-1):
    imName = str(imName)
    assert dtype in [np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
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
                depth = cv2.resize(depth, (im_width, im_height), interpolation=cv2.INTER_AREA )

        depth = np.squeeze(depth)

    return depth
