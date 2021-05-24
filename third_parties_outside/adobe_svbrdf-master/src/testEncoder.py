from torch.optim import Adam
import torch.nn as nn
from loss import *
from render import *
from util import *
import global_var
from torchvision import transforms
import models
import dataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
sys.path.insert(1, 'higan/models/')
from stylegan2_generator import StyleGAN2Generator
import torch.nn.functional as F
import timeit

# import torch
# import numpy as np
# from torch.autograd import Variable
# import torch.optim as optim
# import argparse
# import random
# import os
# import torchvision.utils as vutils
#import utils
# import torch.nn as nn
# import torch.nn.functional as F


np.set_printoptions(precision=4, suppress=True)

# th.autograd.set_detect_anomaly(True)


def save_args(args, dir):
    with open(dir, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def loadLightAndCamera(in_dir):
    print('Load camera position from ', os.path.join(in_dir, 'camera_pos.txt'))
    camera_pos = np.loadtxt(os.path.join(
        in_dir, 'camera_pos.txt'), delimiter=',').astype(np.float32)

    print('Load light position from ', os.path.join(in_dir, 'light_pos.txt'))
    light_pos = np.loadtxt(os.path.join(
        in_dir, 'light_pos.txt'), delimiter=',').astype(np.float32)

    im_size = np.loadtxt(os.path.join(in_dir, 'image_size.txt'), delimiter=',')
    im_size = float(im_size)
    light = np.loadtxt(os.path.join(in_dir, 'light_power.txt'), delimiter=',')

    return light_pos, camera_pos, im_size, light


def loadTarget(in_dir, res, num_render):

    def loadTargetToTensor(dir, res):
        print('Load target image from ', dir)
        target = Image.open(dir)
        if not target.width == res:
            target = target.resize((res, res), Image.LANCZOS)
        target = gyPIL2Array(target)
        target = th.from_numpy(target).permute(2, 0, 1)
        return target

    rendered = th.zeros(num_render, 3, res, res)
    for i in range(num_render):
        rendered[i, :] = loadTargetToTensor(
            os.path.join(in_dir, '%02d.png' % i), res)
    rendered = rendered.cuda()

    texture_fn = os.path.join(in_dir, 'tex.png')
    if os.path.exists(texture_fn):
        textures, res0 = png2tex(texture_fn)
    else:
        textures = None

    return rendered, textures


def initTexture(init_from, res):
    if init_from == 'random':
        textures_tmp = th.rand(1, 9, res, res)
        textures = textures_tmp.clone()
        textures[:, 0:5, :, :] = textures_tmp[:, 0:5, :, :] * 2 - 1
        textures[:, 5, :, :] = textures_tmp[:, 5, :, :] * 1.3 - 0.3
        textures[:, 6:9, :, :] = textures_tmp[:, 6:9, :, :] * 2 - 1
    else:
        textures, _ = png2tex(init_from)
        if res != textures.shape[-1]:
            print('The loaded initial texture has a wrong resolution!')
            exit()
    return textures


def initLatent(genObj, type, init_from):
    if init_from == 'random':
        if type == 'z':
            latent = th.randn(1, 512).cuda()
        elif type == 'w':
            latent = th.randn(1, 512).cuda()
            latent = genObj.net.mapping(latent)
        elif type == 'w+':
            latent = th.randn(1, 512).cuda()
            latent = genObj.net.mapping(latent)
            latent = genObj.net.truncation(latent)
        else:
            print('--gan_latent_type should be z|w|w+')
            exit()
    else:
        if os.path.exists(init_from):
            latent = th.load(init_from).cuda()
        else:
            print('Can not find latent vector ', init_from)
            exit()

    return latent


def updateTextureFromLatent(genObj, type, latent):
    if type == 'z':
        latent = genObj.net.mapping(latent)
        latent = genObj.net.truncation(latent)
    elif type == 'w':
        latent = genObj.net.truncation(latent)
    elif type == 'w+':
        pass

    textures = genObj.net.synthesis(latent)
    textures_tmp = textures.clone()
    textures_tmp[:, 0:5, :, :] = textures[:, 0:5, :, :].clamp(-1, 1)
    textures_tmp[:, 5, :, :] = textures[:, 5, :, :].clamp(-0.3, 1)
    textures_tmp[:, 6:9, :, :] = textures[:, 6:9, :, :].clamp(-1, 1)

    return textures_tmp


def loadBasecolor(path):
    img = Image.open(path)
    img = img.convert('RGB')
    Tsfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])
    img = Tsfm(img)  # 3 x 256 x 256
    return img.cuda()


def tex2pngConvert(tex, fn, isVertical=False):
    # isSpecular = False
    # if tex.size(1) == 9:
    #     isSpecular = True

    albedo, normal, rough, specular = tex2map(tex)  # (x + 1) / 2) ** 2.2

    albedo = gyTensor2Array(albedo[0, :].permute(1, 2, 0))
    normal = gyTensor2Array((normal[0, :].permute(1, 2, 0)+1)/2)
    rough = gyTensor2Array(rough[0, :].permute(1, 2, 0))
    specular = gyTensor2Array(specular[0, :].permute(1, 2, 0))

    # print(np.max(albedo + specular))
    basecolor = (albedo + specular) + \
        np.sqrt((albedo + specular)**2 - 0.16 * albedo)
    # #metallic = np.zeros(basecolor[:,:,0].shape)
    # metallicList = []
    def toLinGray(rgbImg): return 0.2126 * \
        rgbImg[:, :, 0] + 0.7152 * rgbImg[:, :, 1] + 0.0722 * rgbImg[:, :, 2]
    # for c in range(3):
    #     basecolorC = basecolor[:,:,c].reshape(-1)
    #     nonzeroMask = basecolorC > 0
    #     metallicC = np.zeros(basecolorC.shape)
    #     metallicC[nonzeroMask] = (albedo[:,:,c].reshape(-1) / basecolorC)[nonzeroMask]
    #     #metallic += metallicC.reshape(metallic.shape)
    #     metallicList.append(metallicC.reshape(basecolor[:,:,0].shape))
    # metallic = np.stack(metallicList, axis=2)
    # metallic = np.min(metallic, axis=2)
    #metallic = metallic / 3
    basecolorGray = toLinGray(basecolor)
    albedoGray = toLinGray(albedo)
    nonzeroMask = basecolorGray > 0
    metallic = np.zeros(albedoGray.shape)
    metallic[nonzeroMask] = (albedoGray / basecolorGray)[nonzeroMask]

    metallic = 1.0 - metallic
    print(np.max(metallic), np.min(metallic))
    albedo2 = basecolor * (1.0 - metallic[:, :, np.newaxis])
    specular2 = 0.04 * \
        (1.0 - metallic[:, :, np.newaxis]) + \
        basecolor * metallic[:, :, np.newaxis]

    albedo = gyArray2PIL(gyApplyGamma(albedo, 1/2.2))
    normal = gyArray2PIL(normal)
    rough = gyArray2PIL(gyApplyGamma(rough, 1/2.2))
    specular = gyArray2PIL(gyApplyGamma(specular, 1/2.2))
    basecolor = gyArray2PIL(gyApplyGamma(np.clip(basecolor, 0.0, 1.0), 1/2.2))
    #metallic  = gyArray2PIL(gyApplyGamma(metallic, 1/2.2))
    metallic = gyArray2PIL(metallic)
    albedo2 = gyArray2PIL(gyApplyGamma(albedo2, 1/2.2))
    specular2 = gyArray2PIL(gyApplyGamma(specular2, 1/2.2))

    if isVertical:
        png = gyConcatPIL_v(gyConcatPIL_v(albedo, specular), normal)
        png = gyConcatPIL_v(png, rough)
        png = gyConcatPIL_v(gyConcatPIL_v(png, basecolor), metallic)
        png = gyConcatPIL_v(gyConcatPIL_v(png, albedo2), specular2)
    else:
        png = gyConcatPIL_h(gyConcatPIL_h(albedo, specular), normal)
        png = gyConcatPIL_h(png, rough)
        png = gyConcatPIL_h(gyConcatPIL_h(png, basecolor), metallic)
        png = gyConcatPIL_h(gyConcatPIL_h(png, albedo2), specular2)

    if fn is not None:
        png.save(fn)
    return png


def saveTex(tex, save_dir, tmp_dir, idx):
    print('save_dir is %s' % save_dir)
    print('tmp_dir is %s' % tmp_dir)
    fn = os.path.join(save_dir, 'tex%02d.png' % idx)
    #png = tex2pngConvert(tex, fn, isVertical=True)
    png = tex2png(tex, fn, isVertical=False)
    return png


def saveTexAsPNG(tex, save_path):
    png = tex2png(tex, save_path, isVertical=False)
    return png


def renderAndSave(tex, res, size, lp, cp, li, num_render, save_dir, tmp_dir, epoch):
    fn = os.path.join(save_dir, 'tex.png')
    fn2 = os.path.join(save_dir, 'rendered.png')
    png = tex2png(tex, fn)
    # gyCreateThumbnail(fn,128*4,128)

    render_all = None
    for i in range(num_render):
        fn_this = save_dir + '/%02d.png' % i
        render_this = renderTex(
            fn, 256, size, lp[i, :], cp[i, :], li, fn_im=fn_this)
        # gyCreateThumbnail(fn_this)
        render_all = gyConcatPIL_h(render_all, render_this)
        png = gyConcatPIL_h(png, render_this)

    render_all.save(fn2)
    # gyCreateThumbnail(fn2, w=128*num_render, h=128)
    png.save(os.path.join(tmp_dir, 'epoch_%05d.jpg' % epoch))


def convertToSave(matMaps):
    # [batch, 9, H, W] -> [batch, 3, H*4, W]
    def logTensor(val): return (torch.log(val+0.01) - np.log(0.01)) / \
        (np.log(1.01) - np.log(0.01))

    def to3C(img): return torch.cat([img, img, img], dim=1)
    normalXY = matMaps[:, :2, :, :]
    normalZ = torch.sqrt(
        1 - torch.sum(normalXY * normalXY, dim=1, keepdim=True))
    normal = torch.cat([normalXY, normalZ], dim=1) * 0.5 + 0.5
    diffuse = matMaps[:, 2:5, :, :] * 0.5 + 0.5
    diffuse = logTensor(diffuse)
    rough = matMaps[:, 5:6, :, :] * 0.5 + 0.5
    rough = to3C(rough)
    specular = matMaps[:, 6:9, :, :] * 0.5 + 0.5
    specular = logTensor(specular)
    return torch.cat([normal, diffuse, rough, specular], dim=2)


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

def test(opt):
    if opt.experiment is None:
        opt.experiment = 'check_%s_cw%d_sw%d_bn%d' % (
            opt.mode, opt.classWeight, opt.scaleWeight, opt.batchSize)
    os.system('mkdir {0}'.format(opt.experiment))
    #os.system('cp *.py %s' % opt.experiment)

    opt.seed = 0
    print("Random Seed: ", opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    th.manual_seed(opt.seed)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    now = datetime.now()
    print(now)

    imgSavePath = os.path.join(opt.experiment, 'imgs_testEpoch%d' % opt.epochId)
    if not os.path.exists(imgSavePath):
        os.system('mkdir -p {}'.format(imgSavePath))
    modelSavePath = os.path.join(opt.experiment, 'models')

    # Initial Network
    if opt.mode == 'cs':
        net = models.netCS().cuda()
        net = nn.DataParallel(net, device_ids=opt.deviceIds)
        net.load_state_dict(torch.load(os.path.join(modelSavePath, 'net_epoch_{}.pth'.format(opt.epochId)) ) )
        print('Model %s is loaded!' % os.path.join(modelSavePath, 'net_epoch_{}.pth'.format(opt.epochId)) )

    brdfDataset = dataLoader.BatchLoader(opt.dataRoot, opt.matDataRoot,
                                         imWidth=opt.imWidth, imHeight=opt.imHeight, mode=opt.mode,
                                         phase='TEST', split='test')
    brdfLoader = DataLoader(brdfDataset, batch_size=opt.batchSize,
                            num_workers=8, shuffle=True)

    matG1IdDict = getG1IdDict(os.path.join(opt.dataRoot, 'matIdGlobal1.txt'))

    # Loss function
    ceLoss = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    l2Loss = nn.MSELoss()

    correct = 0
    total = 0
    classErrTotal = 0
    scaleErrTotal = 0
    startTime = timeit.default_timer()
    testingLog = open(
        '{0}/testingLog_{1}.txt'.format(opt.experiment, opt.epochId), 'w')
    for i, dataBatch in enumerate(brdfLoader):
        #print('1. time : {}'.format(timeit.default_timer() - startTime) )
        # Load the image from cpu to gpu
        imBatch = Variable(dataBatch['im']).cuda()
        depthBatch = Variable(dataBatch['depth']).cuda()
        matMBatch = Variable(dataBatch['matMask']).cuda()

        if opt.mode == 'cs':
            matLabelBatch = dataBatch['matLabel'].cuda()
            matScaleBatch = dataBatch['matScale'].cuda()
       
        #print('2. time : {}'.format(timeit.default_timer() - startTime) )

        # Initial Prediction
        inputBatch = torch.cat([imBatch, depthBatch, matMBatch.to(torch.float32)], dim=1)
        output = net(inputBatch)

        # Compute the error
        classErr = ceLoss(output['material'], matLabelBatch)
        scaleErr = l2Loss(output['scale'], matScaleBatch)

        #loss = opt.classWeight * classErr + opt.scaleWeight * scaleErr

        _, matLabelPred = torch.max(output['material'].data, 1)
        total += matLabelBatch.size(0)
        correct += (matLabelPred == matLabelBatch).sum().item()
        classErrTotal += (classErr.item() * matLabelBatch.size(0) )
        scaleErrTotal += (scaleErr.item() * matLabelBatch.size(0) )
        # print every 2000 mini-batches
        if i % opt.printStep == (opt.printStep-1) or (i+1) == len(brdfLoader):
            classAccu = correct / total * 100
            classErrCurr = classErrTotal / total
            scaleErrCurr = scaleErrTotal / total
            printMSG('[Epoch %d, %5d/%5d] Class loss: %.4f accu: %.2f %% Scale loss: %.4f ' %
                        (opt.epochId, i + 1, len(brdfLoader), classErrCurr, classAccu, scaleErrCurr), testingLog)

        #print('4. time : {}'.format(timeit.default_timer() - startTime) )
        if i % opt.saveStep == 0:
            to3C = lambda x: torch.cat([x, x, x], dim=1)
            dsF  = lambda x: F.interpolate(x, [x.shape[2] // 2, x.shape[3] // 2], mode='bilinear')
            dsI  = lambda x: F.interpolate(x, [x.shape[2] // 2, x.shape[3] // 2], mode='nearest')
            inputs = torch.cat([dsF(imBatch), to3C(dsF(depthBatch)), to3C(dsI(matMBatch.to(torch.float32) ) )], dim=2)
            
            vutils.save_image(inputs, os.path.join(imgSavePath, 'test_inputs_%04d.png' % i))
            #print('4-1. time : {}'.format(timeit.default_timer() - startTime) )
            matsGT = getRescaledMatFromID(
                matLabelBatch.cpu().numpy(), matScaleBatch.cpu().numpy(), opt.matOriDataRoot, matG1IdDict)
            
            vutils.save_image(matsGT, os.path.join(imgSavePath, 'test_matGT_%04d.png' % i ))
            #print('4-2. time : {}'.format(timeit.default_timer() - startTime) )
            
            matsPred = getRescaledMatFromID(
                matLabelPred.cpu().numpy(), output['scale'].detach().cpu().numpy(), opt.matOriDataRoot, matG1IdDict)
            
            vutils.save_image(matsPred, os.path.join(imgSavePath, 'test_matPred_%04d.png' % i))
            #print('4-3. time : {}'.format(timeit.default_timer() - startTime) )
            albedoBatch = dataBatch['albedo'] ** (1.0/2.2)
            normalBatch = (dataBatch['normal'] + 1) * 0.5
            roughBatch = (dataBatch['rough'] + 1) * 0.5
            matsScene = torch.cat([dsF(albedoBatch), dsF(normalBatch), to3C(dsF(roughBatch) )], dim=2)
            
            vutils.save_image(matsScene, os.path.join(imgSavePath, 'test_matScene_%04d.png' % i))

if __name__ == '__main__':
    print('\n\n\n')

    parser = argparse.ArgumentParser(description='PyTorch Optimization -- GY')
    # The locationi of training set
    parser.add_argument('--dataRoot', default=None,
                        help='path to input images')
    parser.add_argument('--matDataRoot', default=None,
                        help='path to material database')
    parser.add_argument('--matOriDataRoot', default=None,
                        help='path to material database')
    parser.add_argument('--experiment', default=None,
                        help='the path to store samples and models')
    # The basic training setting
    parser.add_argument('--mode', type=str, default='w+n',
                        help='cs for classifier+scale, w and w+ for style, w+n for style and noise')
    # parser.add_argument('--nEpoch', type=int, default=25,
    #                     help='the number of epochs for training')
    parser.add_argument('--epochId', type=int, default=25,
                        help='the epoch ID for testing')
    parser.add_argument('--batchSize', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--imHeight', type=int, default=240)
    parser.add_argument('--imWidth', type=int, default=320)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--deviceIds', type=int, nargs='+',
                        default=[0], help='the gpus used for training network')
    # The training weight
    parser.add_argument('--classWeight', type=float, default=1.0,
                        help='the weight for the diffuse component')
    parser.add_argument('--scaleWeight', type=float, default=1.0,
                        help='the weight for the diffuse component')

    parser.add_argument('--printStep', type=int, default=10, help='print for every # of minibatch')
    parser.add_argument('--saveStep' , type=int, default=100, help='print for every # of minibatch')

    # parser.add_argument('--in_dir', required=True)
    # parser.add_argument('--mat_fn', type=str, default='')
    # parser.add_argument('--out_dir', required=True)
    # parser.add_argument('--vgg_weight_dir', required=True)
    # parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--im_res', type=int, default=256)
    # parser.add_argument('--epochs', type=int, default=1000)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--optim_latent', action='store_true')
    # parser.add_argument('--tex_init', type=str, default='random')
    # parser.add_argument('--num_render', type=int, default=9)
    # parser.add_argument('--num_render_used', type=int, default=5)
    # parser.add_argument('--gan_latent_type', type=str, default='w+')
    # parser.add_argument('--gan_latent_init', type=str, default='random')
    # parser.add_argument('--gan_noise_init', type=str, default='random')
    # parser.add_argument('--sub_epochs', type=int, nargs='+')
    # parser.add_argument('--loss_weight', type=float, nargs='+')
    # parser.add_argument('--seed', type=int, default=None)
    # parser.add_argument('--embed_tex', action='store_true')
    # parser.add_argument('--jittering', action='store_true')
    # parser.add_argument('--applyMask', action='store_true')
    # parser.add_argument('--applyMask2', action='store_true', help='full albedo, masked others')
    # parser.add_argument('--diffuseOnly', action='store_true')
    # parser.add_argument('--alignPixMean', action='store_true')
    # parser.add_argument('--alignPixStd', action='store_true')
    # parser.add_argument('--alignVGG', action='store_true')
    # parser.add_argument('--findInit', action='store_true')

    args = parser.parse_args()

    # # ours
    # # args.vgg_layer_weight_w  = [0.125, 0.125, 0.125, 0.125]
    # # args.vgg_layer_weight_n  = [0.125, 0.125, 0.125, 0.125]

    # args.vgg_layer_weight_w  = [0.071, 0.071, 0.286, 0.572]
    # args.vgg_layer_weight_n  = [0.421, 0.421, 0.105, 0.053]

    # # args.vgg_layer_weight_w  = [0.071, 0.071, 0.286, 0.572]
    # # args.vgg_layer_weight_n  = [0.071, 0.071, 0.286, 0.572]

    # # args.vgg_layer_weight_w  = [0.421, 0.421, 0.105, 0.053]
    # # args.vgg_layer_weight_n  = [0.421, 0.421, 0.105, 0.053]

    # # args.vgg_layer_weight_w  = [0.421, 0.421, 0.105, 0.053]
    # # args.vgg_layer_weight_n  = [0.071, 0.071, 0.286, 0.572]

    # args.vgg_layer_weight_wn = [0.125, 0.125, 0.125, 0.125]

    # if args.sub_epochs[0] == 0:
    #     if args.sub_epochs[1] == 0:
    #         args.optim_strategy = 'L+N'
    #     else:
    #         args.optim_strategy = 'N'
    # else:
    #     if args.sub_epochs[1] == 0:
    #         args.optim_strategy = 'L'
    #     else:
    #         args.optim_strategy = 'L|N'

    # if args.seed:
    #     pass
    # else:
    #     args.seed = random.randint(0, 2**31 - 1)

    # args.epochs = max(args.sub_epochs[0] + args.sub_epochs[1], args.epochs)

    assert args.mode in ['cs', 'w', 'w+', 'w+n']

    test(args)
