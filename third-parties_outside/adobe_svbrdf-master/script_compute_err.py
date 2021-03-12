import sys
sys.path.insert(1, 'src/')
from util import *
from descriptor import FeatureLoss, StyleLoss
from torchvision.transforms import Normalize
sys.path.insert(1, 'PerceptualSimilarity/')
import models

def normalize_vgg19(input):
    transform = Normalize(
        mean=[0.48501961, 0.45795686, 0.40760392],
        std=[1./255, 1./255, 1./255]
    )
    return transform(input)

def loadPng2Tensor(fn):
    im = Image.open(fn)
    im = im.resize((256, 256), Image.LANCZOS)
    im = gyPIL2Array(im)
    im = th.from_numpy(im).permute(2,0,1)
    return im

def loadImageForVGG(fn1, fn2):
    im0 = loadPng2Tensor(fn1)
    im1 = loadPng2Tensor(fn2)
    im  = th.zeros(2, 3, 256, 256)
    im[0,:] = normalize_vgg19(im0)
    im[1,:] = normalize_vgg19(im1)
    return im.cuda()

def loadImage(fn):
    im = loadPng2Tensor(fn)
    im = im.unsqueeze(0)
    return im.cuda()

def LPIPSLoss(LPIPS, fn1, fn2):
    img1 = loadImage(fn1)
    img2 = loadImage(fn2)

    loss_lpips  = LPIPS.forward(img1, img2).sum()
    return loss_lpips

if __name__ == '__main__':
    vgg_weight_dir = 'data/pretrain/vgg_conv.pt'
    FL = FeatureLoss(vgg_weight_dir, [0.125, 0.125, 0.125, 0.125])
    for p in FL.parameters():
        p.requires_grad = False

    criterion = th.nn.MSELoss().cuda()
    LPIPS = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

    pre = '/home/guoyu/Documents/3_svbrdf/svbrdf_paper/materialgan/images/'
    fn1 = pre + 'results/main/real_giftbag1/ref/08.png'
    fn2 = pre + 'results/main/real_giftbag1/msra7_egsr7_refine/08.png'

    loss_lpips = LPIPSLoss(LPIPS, fn1, fn2)
    print(loss_lpips.item())

        # ref = loadImageForVGG(ref_dir + mat + '/07.png', ref_dir + mat + '/08.png')
        # in1 = loadImageForVGG(in1_dir + mat + '/07.png', in1_dir + mat + '/08.png')

        # vgg_ref = FL(ref)
        # vgg_in1 = FL(in1)

        # loss_vgg = criterion(vgg_ref, vgg_in1)

    # np.savetxt(in1_dir + mat + '/error.tex', np.vstack(loss_list_all), fmt='%.4f', delimiter=',')
