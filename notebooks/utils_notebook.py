import numpy as np
from skimage.measure import block_reduce 
import os.path as osp
import numpy as np
import cv2
from skimage.measure import block_reduce 
from PIL import Image


def loadEnvmap(envName, envHeight = 8, envWidth = 16, envRow = 120, envCol = 160, SGNum = 12):
    print('>>>>loadEnvmap', envName)
    if not osp.isfile(envName ):
        env = np.zeros( [3, envRow, envCol,
            envHeight, envWidth], dtype = np.float32 )
        envInd = np.zeros([1, 1, 1], dtype=np.float32 )
        print('Warning: the envmap %s does not exist.' % envName )
        return env, envInd
    else:
        envHeightOrig, envWidthOrig = 16, 32
        assert( (envHeightOrig / envHeight) == (envWidthOrig / envWidth) )
        assert( envHeightOrig % envHeight == 0)

        env = cv2.imread(envName, -1 ) 
        print(env.shape)

        if not env is None:
            env = env.reshape(envRow, envHeightOrig, envCol,
                envWidthOrig, 3) # (1920, 5120, 3) -> (120, 16, 160, 32, 3)
            env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3] ) ) # -> (3, 120, 160, 16, 32)

            scale = envHeightOrig / envHeight
            if scale > 1:
                env = block_reduce(env, block_size = (1, 1, 1, 2, 2), func = np.mean )

            envInd = np.ones([1, 1, 1], dtype=np.float32 )
            return env, envInd
        else:
            env = np.zeros( [3, envRow, envCol,
                envHeight, envWidth], dtype = np.float32 )
            envInd = np.zeros([1, 1, 1], dtype=np.float32 )
            print('Warning: the envmap %s does not exist.' % envName )
            return env, envInd

def loadImage(imName, isGama = False, im_width = 320, im_height = 240):
    if not(osp.isfile(imName ) ):
#         self.logger.warning('File does not exist: ' + imName )
        assert(False )

    im = Image.open(imName)
    im = im.resize([im_width, im_height], Image.ANTIALIAS )

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

def loadHdr(imName, im_width = 320, im_height = 240):
    if not(osp.isfile(imName ) ):
#         print(imName )
        assert(False )
    im = cv2.imread(imName, -1)
    # print(imName, im.shape, im.dtype)

    if im is None:
        print(imName )
        assert(False )
    im = cv2.resize(im, (im_width, im_height), interpolation = cv2.INTER_AREA )
    im = np.transpose(im, [2, 0, 1])
    im = im[::-1, :, :]
    return im.transpose(1, 2, 0)

import struct
def loadBinary(imName, channels = 1, dtype=np.float32, if_resize=True, im_width = 320, im_height = 240):
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
            # print(self.im_width, self.im_height, width, height)
            if dtype == np.float32:
                depth = cv2.resize(depth, (im_width, im_height), interpolation=cv2.INTER_AREA )
            elif dtype == np.int32:
                depth = cv2.resize(depth.astype(np.float32), (im_width, im_height), interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.int32)

        depth = np.squeeze(depth)

    return depth[np.newaxis, :, :]
