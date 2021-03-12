import numpy as np
from PIL import Image

def gyConcatPIL_h(im1, im2):
    if im1 is None:
        return im2
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def gyArray2PIL(im):
    return Image.fromarray((im*255).astype(np.uint8))

def gyApplyGamma(im, gamma):
    if gamma < 1: im = im.clip(min=1e-8)
    return im**gamma

res = 256
albedo = 0.5 * np.ones((256,256,3), dtype=np.float32)
normal_x = 0.5 * np.ones((256,256), dtype=np.float32)
normal_y = 0.5 * np.ones((256,256), dtype=np.float32)
normal_z = np.ones((256,256), dtype=np.float32)
normal = np.stack((normal_x, normal_y, normal_z), axis=2)
rough = 0.2 * np.ones((256,256,3), dtype=np.float32)
spec = 0.04 * np.ones((256,256,3), dtype=np.float32)

albedo = gyArray2PIL(gyApplyGamma(albedo,1/2.2))
normal = gyArray2PIL(normal)
rough = gyArray2PIL(gyApplyGamma(rough,1/2.2))
spec = gyArray2PIL(gyApplyGamma(spec,1/2.2))

tex = gyConcatPIL_h(gyConcatPIL_h(gyConcatPIL_h(albedo,normal), rough), spec)
tex.save('const_init_tex.png')



