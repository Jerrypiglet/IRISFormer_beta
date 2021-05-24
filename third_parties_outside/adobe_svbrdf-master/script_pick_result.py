import sys
sys.path.insert(1, 'src/')
from util import *
from descriptor import FeatureLoss, StyleLoss
from torchvision.transforms import Normalize

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
    imL2  = th.zeros(2, 3, 256, 256)
    imVGG  = th.zeros(2, 3, 256, 256)
    imL2[0,:] = im0
    imL2[1,:] = im1
    imVGG[0,:] = normalize_vgg19(im0)
    imVGG[1,:] = normalize_vgg19(im1)
    return imL2.cuda(), imVGG.cuda()

if __name__ == '__main__':
    vgg_weight_dir = 'data/pretrain/vgg_conv.pt'
    FL = FeatureLoss(vgg_weight_dir, [0.125, 0.125, 0.125, 0.125])
    for p in FL.parameters():
        p.requires_grad = False

    criterion = th.nn.MSELoss().cuda()

    ref_dir = 'data/in/'
    # in1_dir = 'data/out/msra7_avg/'
    # in2_dir = 'data/out/msra7_const/'
    # out_dir = 'data/out/msra7_picked/'
    # in1_r_dir = 'data/out/msra7_avg_refine/'
    # in2_r_dir = 'data/out/msra7_const_refine/'
    # out_r_dir = 'data/out/msra7_picked_refine/'
    in1_dir = 'data/out/ours7_avg/'
    in2_dir = 'data/out/ours7_const/'
    out_dir = 'data/out/ours7_picked/'
    gyCreateFolder(out_dir)
    in1_r_dir = 'data/out/ours7_avg_refine/'
    in2_r_dir = 'data/out/ours7_const_refine/'
    out_r_dir = 'data/out/ours7_picked_refine/'
    gyCreateFolder(out_r_dir)
    mat_list = gyListNames(in2_dir+'*')
    for j, mat in enumerate(mat_list):
        if mat == 'real_stone-spec-ground-flake':
        # if 1 == 1:
            print(mat)

            ref_L2, ref_vgg = loadImageForVGG(ref_dir + mat + '/07.png', ref_dir + mat + '/08.png')
            in1_L2, in1_vgg = loadImageForVGG(in1_dir + mat + '/07.png', in1_dir + mat + '/08.png')
            in2_L2, in2_vgg = loadImageForVGG(in2_dir + mat + '/07.png', in2_dir + mat + '/08.png')

            vgg_ref = FL(ref_vgg)
            vgg_in1 = FL(in1_vgg)
            vgg_in2 = FL(in2_vgg)

            loss1_L2 = criterion(ref_L2, in1_L2)
            loss2_L2 = criterion(ref_L2, in2_L2)

            loss1_VGG = criterion(vgg_ref, vgg_in1)
            loss2_VGG = criterion(vgg_ref, vgg_in2)

            # print(loss1_L2, loss2_L2)
            # print(loss1_VGG, loss2_VGG)
            # exit()

            w = [100,0.001]

            loss1 = loss1_L2 * w[0] + loss1_VGG * w[1]
            loss2 = loss2_L2 * w[0] + loss2_VGG * w[1]
            a = loss1.item()
            b = loss2.item()

            # tmp = a
            # a = b
            # b = tmp

            if  a < b:
                cmd = 'cp -r ' + in1_dir + mat + '/ ' + out_dir
                print(cmd)
                os.system(cmd)
                cmd = 'cp -r ' + in1_r_dir + mat + '/ ' + out_r_dir
                print(cmd)
                os.system(cmd)
            else:
                cmd = 'cp -r ' + in2_dir + mat + '/ ' + out_dir
                print(cmd)
                os.system(cmd)
                cmd = 'cp -r ' + in2_r_dir + mat + '/ ' + out_r_dir
                print(cmd)
                os.system(cmd)
