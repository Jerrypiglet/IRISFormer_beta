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
import torchvision.utils as vutils
sys.path.insert(1, 'higan/models/')
from stylegan2_generator import StyleGAN2Generator
import torch.nn.functional as F
import timeit
import xml.etree.ElementTree as ET
import cv2

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

def getMatFromStyle(genObj, matStyles, mode):
    res = 256
    # initialize noise space
    global_var.init_global_noise(res, 'random')
    # initialize latent space
    latent = matStyles
    #latent = initLatent(genObj, args.gan_latent_type, args.gan_latent_init)
    #latent = Variable(latent, requires_grad=True)
    print('Latent vector shape:', latent.shape)
    # GAN generation
    tex = updateTextureFromLatent(genObj, mode, latent)

    albedo, normal, rough, _ = tex2map(tex)
    albedo = albedo ** (1/2.2)
    normal = (normal + 1) / 2
    rough = rough ** (1/2.2)
    mats = torch.cat([albedo, normal, rough], dim=2)

    return mats

def getTexFromStyle(genObj, matStyles, mode='w', noiseSrc='random'):
    # initialize noise space
    global_var.init_global_noise(256, noiseSrc)
    # initialize latent space
    latent = matStyles
    #print('Latent vector shape:', latent.shape)
    # GAN generation
    tex = updateTextureFromLatent(genObj, mode, latent)
    return tex

def renderBatchFrame(tex, tex_res, size, lp, cp, li, nC, nR):
    if tex.size(0) < nC*nR:
        print('batch size is too small!')
        assert(False)
    tex = tex[:,:6,:,:]
    renderObj = Microfacet(res=tex_res, size=size)
    im = renderObj.eval(tex, lightPos=lp[0, :], \
        cameraPos=cp[0, :], light=th.from_numpy(li).cuda())
    #im = gyApplyGamma(gyTensor2Array(im[0,:].permute(1,2,0)), 1/2.2)
    im = im ** (1/2.2)
    albedo, normal, rough = tex2map(tex)
    albedo = albedo ** (1/2.2)
    normal = (normal + 1) / 2
    rough = rough ** (1/2.2)
    frame = th.cat([albedo, normal, rough, im], dim=2) # batch x 3 x 4H x W
    framesR = []
    for fr in range(nR):
        framesC = []
        for fc in range(nC):
            framesC.append(frame[fc+fr*nC])
        frameC = th.cat(framesC, dim=2) # 3 x 4H x bW
        framesR.append(frameC)
    frame = th.cat(framesR, dim=1) # 3 x 4HnR x nCW
    return (frame.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8') # 4H x bW x 3

def saveVideo(frames, outputName="output.avi"):
    ####
    print(frames[0].shape)
    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(outputName, cv2.VideoWriter_fourcc(*"MJPG"), 24,(width,height))

    for frame in frames:
        #writer.write(np.random.randint(0, 255, (480,640,3)).astype('uint8'))
        writer.write(frame.astype('uint8')[:,:,::-1])

    #cv2.destroyAllWindows()
    writer.release()

    Image.fromarray(frames[0]).save(outputName.replace('.avi', '.png'))

def saveMatPred(matsPred, matPredDir, partIdList, partNameDict):
    matSavePathDict = {} # xxx_yyy_partk: [albedoPath, normalPath, roughPath]
    os.system('mkdir -p %s' % matPredDir)
    for idx, partId in enumerate(partIdList):
        matPred = matsPred[idx] # 3 x 3H x W , between [0, 1]
        H = matPred.shape[1] // 3
        albedoPath = os.path.join(matPredDir, '%s_diffuse.png' % partId)
        vutils.save_image(matPred[:, :H, :], albedoPath)
        normalPath = os.path.join(matPredDir, '%s_normal.png' % partId)
        vutils.save_image(matPred[:, H:2*H, :], normalPath)
        roughPath = os.path.join(matPredDir, '%s_rough.png' % partId)
        vutils.save_image(matPred[:, 2*H:, :], roughPath)
        matSavePathDict[partNameDict[partId]] = [albedoPath, normalPath, roughPath]
    return matSavePathDict

def saveNewXml(xmlFileNew, matSavePathDict, isFast=False):
    tree = ET.parse(xmlFileNew)
    root = tree.getroot()
    for child in root: 
        if child.tag == 'bsdf':
            bsdfStrId = child.attrib['id'] # material part name = cadcatID_objID_partID
            if bsdfStrId not in matSavePathDict.keys():
                print('material part: %s not shown in this view, skip!' % bsdfStrId)
                continue
            newDirName = os.path.basename(os.path.dirname(matSavePathDict[bsdfStrId][0]) )
            for child2 in child:
                if child2.tag == 'texture' and child2.attrib['name'] == 'albedo':
                    fn = os.path.basename(matSavePathDict[bsdfStrId][0])
                    child2[0].set('value', os.path.join(newDirName, fn))
                    #child2[0].set('value', matSavePathDict[bsdfStrId][0])
                if child2.tag == 'texture' and child2.attrib['name'] == 'normal':
                    fn = os.path.basename(matSavePathDict[bsdfStrId][1])
                    child2[0].set('value', os.path.join(newDirName, fn))
                    #child2[0].set('value', matSavePathDict[bsdfStrId][1])
                if child2.tag == 'texture' and child2.attrib['name'] == 'roughness':
                    fn = os.path.basename(matSavePathDict[bsdfStrId][2])
                    child2[0].set('value', os.path.join(newDirName, fn))
                    #child2[0].set('value', matSavePathDict[bsdfStrId][2])
                if child2.tag == 'rgb' and child2.attrib['name'] == 'albedoScale':
                    child2.set('value', '1.000 1.000 1.000')
                if child2.tag == 'float' and child2.attrib['name'] == 'roughnessScale':
                    child2.set('value', '1.000')
        if child.tag == 'sensor' and isFast:
            for child2 in child:
                if child2.tag == 'sampler':
                    child2.set('type', 'independent')
    tree.write(xmlFileNew)

def test(opt):
    opt.seed = 0
    print("Random Seed: ", opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    th.manual_seed(opt.seed)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    now = datetime.now()
    print(now)

    imgSavePath = os.path.join(opt.sampleSavePath, 'sampleW')
    if not os.path.exists(imgSavePath):
        os.system('mkdir -p {}'.format(imgSavePath))

    # Initial Network
    genObj = StyleGAN2Generator('svbrdf')
    
    brdfDataset = dataLoader.MatWLoader(opt.matDataRoot)
    brdfLoader = DataLoader(brdfDataset, batch_size=opt.batchSize,
                            num_workers=8, shuffle=True)

    startTime = timeit.default_timer()
    in_this_dir = os.path.join('data', 'in', 'real_plastic-red-carton')
    light_pos, camera_pos, im_size, light = loadLightAndCamera(in_this_dir)
    for i, dataBatch in enumerate(brdfLoader):
        # Load the image from cpu to gpu
        matStyleBatch = dataBatch['matStyle'].cuda()
        matNoisePathBatch = dataBatch['matNoisePath']

        frames = []
        for n in range(opt.nNoise):
            print('n: %d' % n)
            
            #frames.append(frame)
            texRnd = []
            texOri = []
            for b in range(matStyleBatch.size(0)):
                tex = getTexFromStyle(genObj, matStyleBatch.detach()[b].unsqueeze(0), 'w')
                texRnd.append(tex)
                print('b: %d' % b)
                tex2 = getTexFromStyle(genObj, matStyleBatch.detach()[b].unsqueeze(0), 'w', matNoisePathBatch[b])
                texOri.append(tex2)
            texRnd = torch.cat(texRnd, dim=0)
            frame = renderBatchFrame(texRnd, 256, im_size, light_pos, camera_pos, light, opt.nC, opt.nR)
            texOri = torch.cat(texOri, dim=0)
            frameOri = renderBatchFrame(texOri, 256, im_size, light_pos, camera_pos, light, opt.nC, opt.nR)
            frames.append(np.concatenate([frame, frameOri], axis=0)) # stack vertically
        saveVideo(frames, os.path.join(imgSavePath, '%d.avi' % (i+1) ) )

        if i == opt.nOutput-1:
            break
               # vutils.save_image(matsPred, os.path.join(imgSavePath, 'eval_matPred_%04d.png' % i))

if __name__ == '__main__':
    print('\n\n\n')

    parser = argparse.ArgumentParser(description='PyTorch Optimization -- GY')
    # The locationi of training set
    parser.add_argument('--matDataRoot', default=None,
                        help='path to material database')
    parser.add_argument('--sampleSavePath', default=None)
    parser.add_argument('--nNoise', type=int, default=24, help='for each style vector, sample n times noise vector')
    parser.add_argument('--nOutput', type=int, default=10, help='save n videos')
    # The basic training setting
    parser.add_argument('--batchSize', type=int,
                        default=8, help='input batch size')
    parser.add_argument('--nC', type=int,
                        default=8, help='output frame columns')
    parser.add_argument('--nR', type=int,
                        default=1, help='output frame rows')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--deviceIds', type=int, nargs='+',
                        default=[0], help='the gpus used for training network')

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    test(args)
