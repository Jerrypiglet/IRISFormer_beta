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
from tqdm import tqdm
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
    mats = torch.cat([albedo, normal, rough], dim=2) # 1 x 3 x 3H x W

    return mats

def getMeanMatFromIRNet(matMBatch, predRoot, viewId, res=128):
    h, w = matMBatch.size(2), matMBatch.size(3)
    #items = ['albedoBS1.png', 'roughBS1.png']
    albedoFile = os.path.join(predRoot, '%s_albedoBS1.png' % viewId)
    albedoOri = Image.open(albedoFile).convert('RGB').resize((w, h), Image.ANTIALIAS)
    albedoOri = np.asarray(albedoOri) / 255.0
    albedoOri = np.reshape(albedoOri, (h*w, -1))
    roughFile = os.path.join(predRoot, '%s_roughBS1.png' % viewId)
    roughOri  = Image.open(roughFile).convert('L').resize((w, h), Image.ANTIALIAS)
    roughOri = np.asarray(roughOri) / 255.0
    roughOri = np.reshape(roughOri, (h*w))
    mats = []
    for i in range(matMBatch.size(0)):
        #mask = matMBatch[i].squeeze().unsqueeze(2).view(-1)
        mask = matMBatch[i].squeeze().cpu().numpy().reshape((h*w)) > 0.9
        rgb = np.mean(albedoOri[mask], axis=0)
        #rgb = np.clip(rgb, 0.0, 1.0) ** (1/2.2)
        rgb = np.clip(rgb, 0.0, 1.0)
        albedo = np.tile(rgb, (res, res, 1)) # res x res x 3
        normal = np.tile(np.array([0.5, 0.5, 1]), [res, res, 1]) # res x res x 3
        rough = np.mean(roughOri[mask])
        rough = np.clip(float(rough), 0.0, 1.0)
        rough = np.tile(rough, (res, res, 3)) # res x res x 3
        mat = np.concatenate([albedo, normal, rough], axis=0) # 3res x res x 3
        mats.append(th.from_numpy(np.transpose(mat, [2, 0, 1]).astype(np.float32))) # 3 x 3res x res

    return mats

def getOptimFromCrop(imBatch, matMBatch, predRoot, viewId, res=128):
    h, w = matMBatch.size(2), matMBatch.size(3)
    normalFile = os.path.join(predRoot, '%s_normal1.png' % viewId)
    normalOri  = Image.open(normalFile).convert('RGB').resize((w, h), Image.ANTIALIAS)
    normalOri  = np.asarray(normalOri) / 255.0
    normalOri  = normalOri * 2 - 1
    normalOri  = np.reshape(normalOri, (h*w, -1))
    imFile     = os.path.join(predRoot, '%s.png' % viewId)
    imOri      = Image.open(normalFile).convert('RGB')
    imResized  = imOri.resize((w, h), Image.ANTIALIAS)
    imOri      = (np.asarray(imOri) / 255.0 ) ** 2.2
    envFile    = os.path.join(predRoot, '%s_envmap1.png' % viewId)
    ######

    ######
    mats = []
    for i in range(matMBatch.size(0)):
        mask = matMBatch[i].squeeze().cpu().numpy() > 0.9
        M = cv2.moments(mask.astype(np.uint8))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print(cX, cY)
        continue
        assert(False)
        mask = mask.reshape((h*w))
        normal = np.mean(normalOri[mask], axis=0)
        planeNormal = normal / np.linalg.norm(normal)
        
        renderObj = MicrofacetPlaneCrop(res, bbox, planeNormal)
    assert(False)

def getOptimFromCropDemo(imBatch, matMBatch, predRoot, viewId, partId, res=128, meanReg=False):
    saveDir = '/siggraphasia20dataset/code/Routine/DatasetCreation/OpenRoomScanNetView/scene0001_00/%s/demo' % viewId
    if meanReg:
        saveDir = saveDir.replace('demo', 'demoReg')

    h, w = matMBatch.size(2), matMBatch.size(3)
    normalFile = os.path.join(predRoot, '%s_normal1.png' % viewId)
    normalOri  = Image.open(normalFile).convert('RGB').resize((w, h), Image.ANTIALIAS)
    normalOri  = np.asarray(normalOri) / 255.0
    normalOri  = normalOri * 2 - 1
    normalOri  = normalOri / np.sqrt(np.sum(normalOri ** 2, axis=2) )[:,:,np.newaxis]
    normalOriFlat = np.reshape(normalOri, (h*w, -1))
    imFile     = os.path.join(predRoot, '%s.png' % viewId)
    imOri      = Image.open(imFile).convert('RGB')
    imResized  = imOri.resize((w, h), Image.ANTIALIAS)
    imOri      = (np.asarray(imOri, dtype=np.float32) / 255.0 ) ** 2.2
    envFile    = os.path.join(predRoot, '%s_envmap1.npy' % viewId)
    envmap     = np.load(envFile) # [bn=1, 3, envRow, envCol, self.envHeight, self.envWidth]

    albedoFile = os.path.join(predRoot, '%s_albedoBS1.png' % viewId)
    albedoOri  = Image.open(albedoFile).convert('RGB').resize((w, h), Image.ANTIALIAS)
    albedoOri  = (np.asarray(albedoOri, dtype=np.float32) / 255.0 ) ** 2.2
    roughFile  = os.path.join(predRoot, '%s_roughBS1.png' % viewId)
    roughOri   = Image.open(roughFile).convert('L').resize((w, h), Image.ANTIALIAS)
    roughOri   = (np.asarray(roughOri, dtype=np.float32) / 255.0 )

    def getMask(bbox, h, w):
        m0 = np.zeros((int(bbox[1]*h), int(w)))
        m10 = np.zeros((int(bbox[3]*h), int(bbox[0]*w)))
        m11 = np.ones((int(bbox[3]*h), int(bbox[2]*w)))
        m12 = np.zeros((int(bbox[3]*h), int((1-bbox[0]-bbox[2])*w)))
        m1 = np.concatenate([m10, m11, m12], axis=1)
        m2 = np.zeros((int((1-bbox[1]-bbox[3])*h), int(w)))
        return np.concatenate([m0, m1, m2], axis=0) > 0.5
    # create a mask for 1/16 bottom right corner
    # m1 = np.zeros((int(0.75*h), int(w)))
    # m2 = np.zeros((int(0.25*h), int(0.75*w)))
    # m3 = np.ones((int(0.25*h), int(0.25*w)))
    # m23 = np.concatenate([m2, m3], axis=1)
    # mask = np.concatenate([m1, m23], axis=0)

    
    #partId = 8
    
    if partId == 8: # floor
        bbox = [0.75, 0.75, 0.25, 0.25] # relC, relR, relW, relH
        partStr = 'floor'
    elif partId == 7:
        bbox = [0.25, 0   , 0.25, 0.20]
        partStr = 'wall'
    elif partId == 1:
        bbox = [0.5 , 0.5 , 0.1 , 0.2 ]
        partStr = 'sofa'
    elif partId == 6:
        bbox = [0.95, 0.40, 0.05 ,0.10]
        partStr = 'door'
    elif partId == 2: # table
        bbox = [0.15, 0.40 , 0.10, 0.10]
        partStr = 'table'
    savePartDir = os.path.join(saveDir, '%d_%s' % (partId, partStr) )
    print('mkdir -p %s' % savePartDir)
    os.system('mkdir -p %s' % savePartDir)
    
    h, w, _ = normalOri.shape
    mask = getMask(bbox, h, w)

    #mask = np.zeros((w, h))
    print(np.sum(mask))
    #mask[int(0.75*h):, int(0.75*w):] = 1
    print(np.sum(mask), w, h)
    # M = cv2.moments(mask.astype(np.uint8))
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    # print(cX, cY)
    mask = mask.reshape((h*w))
    #print(normalOriFlat[mask].shape)
    normal = np.mean(normalOriFlat[mask], axis=0)
    planeNormal = normal / np.linalg.norm(normal)
    #planeNormal = normal / np.sqrt(np.sum(normal ** 2, axis=2) )[:,:,np.newaxis]
    print('BBox:', bbox, '; Plane Normal:', planeNormal)
    

    ######### Below for debugging ############
    # # load pre-computed materials
    # demoMatDir = '/siggraphasia20dataset/code/Routine/DatasetCreation/OpenRoomScanNetView/scene0001_00/imscannetirmmapmatPred_14'
    # alb = (np.asarray(Image.open(os.path.join(demoMatDir, '%d_diffuse.png' % partId)).convert('RGB') ) / 255.0) ** 2.2
    # nor = np.asarray(Image.open(os.path.join(demoMatDir, '%d_normal.png' % partId)).convert('RGB') ) / 255.0 * 2 - 1
    # rou = (np.asarray(Image.open(os.path.join(demoMatDir, '%d_rough.png' % partId)).convert('RGB') ) / 255.0)
    # textures = [alb[np.newaxis, :, :, :].transpose(0, 3, 1, 2), nor[np.newaxis, :, :, :].transpose(0, 3, 1, 2), rou[np.newaxis, :, :, :].transpose(0, 3, 1, 2)]
    # im = renderObj.eval(textures, envmap, isDemo=True)
    # im = gyApplyGamma(gyTensor2Array(im[0,:].permute(1,2,0)), 1/2.2)
    # im = gyArray2PIL(im)
    # im.save('/siggraphasia20dataset/code/Routine/DatasetCreation/OpenRoomScanNetView/scene0001_00/crop_render.png')
    # print('image saved at %s' % '/siggraphasia20dataset/code/Routine/DatasetCreation/OpenRoomScanNetView/scene0001_00/crop_render.png')
    ######### Above for debugging ############

    imH, imW, imC = imOri.shape
    imCrop = imOri[int(imH*bbox[1]):int(imH*(bbox[1]+bbox[3])), int(imW*bbox[0]):int(imW*(bbox[0]+bbox[2])), :]
    imCrop = gyApplyGamma(imCrop, 1/2.2)
    target_ref = th.nn.functional.interpolate(th.from_numpy(imCrop).unsqueeze(0).permute(0, 3, 1, 2), size=(res, res) )
    target_ref = target_ref.cuda()
    imCrop = gyArray2PIL(imCrop)
    imCrop.save('%s/crop_imgOri.png' % savePartDir)
    print('image saved at %s' % ('%s/crop_imgOri.png' % savePartDir) )

    def cropArrAndResize(inp, bbox, res, planeNormal=None):
        print(inp.shape)
        H, W, C = inp.shape
        crop = inp[int(H*bbox[1]):int(H*(bbox[1]+bbox[3])), int(W*bbox[0]):int(W*(bbox[0]+bbox[2])), :].astype(np.float32)
        crop = th.nn.functional.interpolate(th.from_numpy(crop).unsqueeze(0).permute(0, 3, 1, 2), size=(res, res) )
        if planeNormal is not None:
            #print(th.mean(crop[0].view(3, res*res), dim=1))
            def getRot(a, b): # get rotation matrix from vec a to b
                # a: size 3, b: size 3
                a = a / th.norm(a)
                b = b / th.norm(b)
                nu = th.cross(a, b)
                nu_skew = th.tensor([[0, -nu[2], nu[1]] , 
                                    [nu[2], 0, -nu[0]] , 
                                    [-nu[1], nu[0], 0] ])
                #s = th.norm(nu)
                c = th.dot(a, b)
                R = th.eye(3) + nu_skew + th.matmul(nu_skew, nu_skew) / (1 + c)
                return R
            rot = getRot(th.from_numpy(planeNormal.astype(np.float32)), th.tensor([0.0, 0.0, 1.0]))
            #print(rot, getRot(th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 1.0])),  getRot(th.tensor([0.0, 1.0, 0.0]), th.tensor([0.0, 0.0, 1.0])))
            crop = th.matmul(rot, crop.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2)
            #print(th.mean(crop[0].view(3, res*res), dim=1))
        crop = gyTensor2Array(crop[0,:].permute(1,2,0))
        #crop = gyArray2PIL(crop)
        return crop

    ##### Save Original Crops #####
    alb = cropArrAndResize(albedoOri, bbox, res) ** (1/2.2)
    nor = cropArrAndResize(normalOri, bbox, res, planeNormal = planeNormal)
    #nor = cropArrAndResize(normalOri, bbox, res)
    nor = (nor + 1) * 0.5
    rou = cropArrAndResize(roughOri[:,:,np.newaxis], bbox, res)[:,:,0]
    if meanReg:
        albMean = np.mean(alb ** 2.2, axis=(0, 1))
        rouMean = np.mean(rou)
        oriMeans = {'albedo': albMean, 'rough': rouMean}
    else:
        oriMeans = None

    ms = [alb, nor, rou]
    msName = ['albedo', 'normal', 'rough']
    for i, mm in enumerate(ms):
        im = gyArray2PIL(mm)
        savePath = '%s/crop_%sOri.png' % (savePartDir, msName[i])
        im.save(savePath)
        print('image saved at %s' % savePath)
    ################################

    # optimize latent vectors
    latent_init_from = 'data/pretrain/latent_avg_W_256.pt'
    noise_init_from  = 'random'
    # styles = []
    # noises = []
    # for idx, mat in enumerate(mats):
    #     textures_ref = mat2ref(mat)
    #     #####
    #     print('\n########## Optimizing material [%d/%d] ##########\n' % (idx+1, len(mats)))
    renderObj = MicrofacetPlaneCrop(res, bbox, planeNormal)
    latent_best, noise_best, texture_best, rendered_best = \
        optimLatentRender(args, target_ref, renderObj, envmap, bbox, planeNormal, latent_init_from, noise_init_from, res, oriMeans=oriMeans)
    print(texture_best.size(), rendered_best.size())
    rendered_best = gyApplyGamma(gyTensor2Array(rendered_best[0,:].permute(1,2,0)), 1/2.2)
    rendered_best = gyArray2PIL(rendered_best)
    rendered_best.save('%s/crop_renderOpt.png' % savePartDir)
    print('image saved at %s' % ('%s/crop_renderOpt.png' % savePartDir) )

    # convert to albedo, normal, rough
    albedo, normal, rough, _ = tex2map(texture_best)
    albedo = albedo ** (1/2.2)
    normal = (normal + 1) / 2
    #rough = rough ** (1/2.2)
    mats = torch.cat([albedo, normal, rough], dim=2) # 1 x 3 x 3H x W

    albedo = gyArray2PIL(gyTensor2Array(albedo[0,:].permute(1,2,0)))
    albedoPath = '%s/crop_albedoOpt.png' % savePartDir
    albedo.save(albedoPath)
    print('image saved at %s' % (albedoPath) )
    normal = gyArray2PIL(gyTensor2Array(normal[0,:].permute(1,2,0)))
    normalPath = '%s/crop_normalOpt.png' % savePartDir
    normal.save(normalPath)
    print('image saved at %s' % (normalPath) )
    rough = gyArray2PIL(gyTensor2Array(rough[0,:].permute(1,2,0)))
    roughPath = '%s/crop_roughOpt.png' % savePartDir
    rough.save(roughPath)
    print('image saved at %s' % (roughPath) )
    savePathList = [albedoPath, normalPath, roughPath]

    return mats, savePathList

def getOptimFromCropDemo2(imBatch, matMBatch, predRoot, viewIds, partId, res=128, meanReg=False):
    saveDir = '/siggraphasia20dataset/code/Routine/DatasetCreation/OpenRoomScanNetView/scene0001_00/%s_%s/demo' % (viewIds[0], viewIds[1])
    if meanReg:
        saveDir = saveDir.replace('demo', 'demoReg')
    envmapList = []
    bboxList = []
    planeNormalList = []
    renderObjList = []
    target_refList = []
    for viewId in viewIds:
        h, w = matMBatch.size(2), matMBatch.size(3)
        normalFile = os.path.join(predRoot, '%s_normal1.png' % viewId)
        normalOri  = Image.open(normalFile).convert('RGB').resize((w, h), Image.ANTIALIAS)
        normalOri  = np.asarray(normalOri) / 255.0
        normalOri  = normalOri * 2 - 1
        normalOri  = normalOri / np.sqrt(np.sum(normalOri ** 2, axis=2) )[:,:,np.newaxis]
        normalOriFlat = np.reshape(normalOri, (h*w, -1))
        imFile     = os.path.join(predRoot, '%s.png' % viewId)
        imOri      = Image.open(imFile).convert('RGB')
        imResized  = imOri.resize((w, h), Image.ANTIALIAS)
        imOri      = (np.asarray(imOri, dtype=np.float32) / 255.0 ) ** 2.2
        envFile    = os.path.join(predRoot, '%s_envmap1.npy' % viewId)
        envmap     = np.load(envFile) # [bn=1, 3, envRow, envCol, self.envHeight, self.envWidth]
        envmapList.append(envmap)

        albedoFile = os.path.join(predRoot, '%s_albedoBS1.png' % viewId)
        albedoOri  = Image.open(albedoFile).convert('RGB').resize((w, h), Image.ANTIALIAS)
        albedoOri  = (np.asarray(albedoOri, dtype=np.float32) / 255.0 ) ** 2.2
        roughFile  = os.path.join(predRoot, '%s_roughBS1.png' % viewId)
        roughOri   = Image.open(roughFile).convert('L').resize((w, h), Image.ANTIALIAS)
        roughOri   = (np.asarray(roughOri, dtype=np.float32) / 255.0 )

        def getMask(bbox, h, w):
            m0 = np.zeros((int(bbox[1]*h), int(w)))
            m10 = np.zeros((int(bbox[3]*h), int(bbox[0]*w)))
            m11 = np.ones((int(bbox[3]*h), int(bbox[2]*w)))
            m12 = np.zeros((int(bbox[3]*h), int((1-bbox[0]-bbox[2])*w)))
            m1 = np.concatenate([m10, m11, m12], axis=1)
            m2 = np.zeros((int((1-bbox[1]-bbox[3])*h), int(w)))
            return np.concatenate([m0, m1, m2], axis=0) > 0.5
    
        if partId == 8: # floor
            bboxDict = {'14': [0.75, 0.75, 0.25, 0.25], '154': [0, 0.75, 0.25, 0.25]} # relC, relR, relW, relH
            partStr = 'floor'
        elif partId == 7:
            bboxDict = {'14': [0.25, 0   , 0.25, 0.20]}
            partStr = 'wall'
        elif partId == 1:
            bboxDict = {'14': [0.5 , 0.5 , 0.1 , 0.2 ]}
            partStr = 'sofa'
        elif partId == 6:
            bboxDict = {'14': [0.95, 0.40, 0.05 ,0.10]}
            partStr = 'door'
        elif partId == 2: # table
            bboxDict = {'14': [0.15, 0.40 , 0.10, 0.10]}
            partStr = 'table'
        bbox = bboxDict[viewId]
        bboxList.append(bbox)
        savePartDir = os.path.join(saveDir, '%d_%s' % (partId, partStr) )
        print('mkdir -p %s' % savePartDir)
        os.system('mkdir -p %s' % savePartDir)
        
        h, w, _ = normalOri.shape
        mask = getMask(bbox, h, w)
        mask = mask.reshape((h*w))
        #print(normalOriFlat[mask].shape)
        normal = np.mean(normalOriFlat[mask], axis=0)
        planeNormal = normal / np.linalg.norm(normal)
        planeNormalList.append(planeNormal)
        #planeNormal = normal / np.sqrt(np.sum(normal ** 2, axis=2) )[:,:,np.newaxis]
        print('BBox:', bbox, '; Plane Normal:', planeNormal)
        renderObj = MicrofacetPlaneCrop(res, bbox, planeNormal)
        renderObjList.append(renderObj)

        imH, imW, imC = imOri.shape
        imCrop = imOri[int(imH*bbox[1]):int(imH*(bbox[1]+bbox[3])), int(imW*bbox[0]):int(imW*(bbox[0]+bbox[2])), :]
        imCrop = gyApplyGamma(imCrop, 1/2.2)
        target_ref = th.nn.functional.interpolate(th.from_numpy(imCrop).unsqueeze(0).permute(0, 3, 1, 2), size=(res, res) )
        target_ref = target_ref.cuda()
        target_refList.append(target_ref)
        imCrop = gyArray2PIL(imCrop)
        imCrop.save('%s/%s_crop_imgOri.png' % (savePartDir, viewId) )
        print('image saved at %s' % ('%s/%s_crop_imgOri.png' % (savePartDir, viewId) ) )

        def cropArrAndResize(inp, bbox, res, planeNormal=None):
            print(inp.shape)
            H, W, C = inp.shape
            crop = inp[int(H*bbox[1]):int(H*(bbox[1]+bbox[3])), int(W*bbox[0]):int(W*(bbox[0]+bbox[2])), :].astype(np.float32)
            crop = th.nn.functional.interpolate(th.from_numpy(crop).unsqueeze(0).permute(0, 3, 1, 2), size=(res, res) )
            if planeNormal is not None:
                #print(th.mean(crop[0].view(3, res*res), dim=1))
                def getRot(a, b): # get rotation matrix from vec a to b
                    # a: size 3, b: size 3
                    a = a / th.norm(a)
                    b = b / th.norm(b)
                    nu = th.cross(a, b)
                    nu_skew = th.tensor([[0, -nu[2], nu[1]] , 
                                        [nu[2], 0, -nu[0]] , 
                                        [-nu[1], nu[0], 0] ])
                    #s = th.norm(nu)
                    c = th.dot(a, b)
                    R = th.eye(3) + nu_skew + th.matmul(nu_skew, nu_skew) / (1 + c)
                    return R
                rot = getRot(th.from_numpy(planeNormal.astype(np.float32)), th.tensor([0.0, 0.0, 1.0]))
                #print(rot, getRot(th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 1.0])),  getRot(th.tensor([0.0, 1.0, 0.0]), th.tensor([0.0, 0.0, 1.0])))
                crop = th.matmul(rot, crop.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2)
                #print(th.mean(crop[0].view(3, res*res), dim=1))
            crop = gyTensor2Array(crop[0,:].permute(1,2,0))
            #crop = gyArray2PIL(crop)
            return crop

        ##### Save Original Crops #####
        alb = cropArrAndResize(albedoOri, bbox, res) ** (1/2.2)
        nor = cropArrAndResize(normalOri, bbox, res, planeNormal = planeNormal)
        #nor = cropArrAndResize(normalOri, bbox, res)
        nor = (nor + 1) * 0.5
        rou = cropArrAndResize(roughOri[:,:,np.newaxis], bbox, res)[:,:,0]
        if meanReg:
            albMean = np.mean(alb ** 2.2, axis=(0, 1))
            rouMean = np.mean(rou)
            oriMeans = {'albedo': albMean, 'rough': rouMean}
        else:
            oriMeans = None

        ms = [alb, nor, rou]
        msName = ['albedo', 'normal', 'rough']
        for i, mm in enumerate(ms):
            im = gyArray2PIL(mm)
            savePath = '%s/%s_crop_%sOri.png' % (savePartDir, viewId, msName[i])
            im.save(savePath)
            print('image saved at %s' % savePath)
        ################################

    # optimize latent vectors
    latent_init_from = 'data/pretrain/latent_avg_W_256.pt'
    noise_init_from  = 'random'
    # styles = []
    # noises = []
    # for idx, mat in enumerate(mats):
    #     textures_ref = mat2ref(mat)
    #     #####
    #     print('\n########## Optimizing material [%d/%d] ##########\n' % (idx+1, len(mats)))
    
    latent_best, noise_best, texture_best, rendered_bestList = \
        optimLatentRender2(args, target_refList, renderObjList, envmapList, bboxList, planeNormalList, latent_init_from, noise_init_from, res, oriMeans=oriMeans)
    #print(texture_best.size(), rendered_best.size())
    for idx, viewId in enumerate(viewIds):
        rendered_best = rendered_bestList[idx]
        rendered_best = gyApplyGamma(gyTensor2Array(rendered_best[0,:].permute(1,2,0)), 1/2.2)
        rendered_best = gyArray2PIL(rendered_best)
        rendered_best.save('%s/%s_crop_renderOpt.png' % (savePartDir, viewId) )
        print('image saved at %s' % ('%s/%s_crop_renderOpt.png' % (savePartDir, viewId) ) )

    # convert to albedo, normal, rough
    albedo, normal, rough, _ = tex2map(texture_best)
    albedo = albedo ** (1/2.2)
    normal = (normal + 1) / 2
    #rough = rough ** (1/2.2)
    mats = torch.cat([albedo, normal, rough], dim=2) # 1 x 3 x 3H x W

    albedo = gyArray2PIL(gyTensor2Array(albedo[0,:].permute(1,2,0)))
    albedoPath = '%s/crop_albedoOpt.png' % savePartDir
    albedo.save(albedoPath)
    print('image saved at %s' % (albedoPath) )
    normal = gyArray2PIL(gyTensor2Array(normal[0,:].permute(1,2,0)))
    normalPath = '%s/crop_normalOpt.png' % savePartDir
    normal.save(normalPath)
    print('image saved at %s' % (normalPath) )
    rough = gyArray2PIL(gyTensor2Array(rough[0,:].permute(1,2,0)))
    roughPath = '%s/crop_roughOpt.png' % savePartDir
    rough.save(roughPath)
    print('image saved at %s' % (roughPath) )
    savePathList = [albedoPath, normalPath, roughPath]

    return mats, savePathList

def mat2MeanScales(mat): # 3 x 3H x W --> r,g,b,rough values
    mat = mat.permute(1, 2, 0) # 3H x W x 3 (assume already in linear space)
    H = mat.shape[0] // 3
    albedo = mat[:H, :, :] # H x W x 3
    albedoVal = th.mean(albedo, dim=(0, 1))
    rough  = mat[2*H:, :, :] # H x W x 3
    roughVal = th.mean(rough)
    return albedoVal, roughVal # 

def rescaleToMean(mats, matsMean): # mats: nMat x 3 x 3H x W
    newMats = []
    #for idx, mat in enumerate(mats): # 3 x 3H x W
    for idx in range(mats.size(0)):
        mat = mats[idx]
        print(th.max(mat), th.min(mat), mat.shape)
        rgbMeanScale, roughMeanScale = mat2MeanScales(matsMean[idx] ** 2.2)
        H = mat.shape[1] // 3
        rgbOriScale = th.mean(mat[:, :H, :] ** 2.2, dim=(1, 2))
        rgbScale    = rgbMeanScale.cuda() / th.maximum(rgbOriScale, th.ones_like(rgbOriScale) * 1e-8)
        print(idx)
        print(rgbMeanScale)
        print(th.mean((mat[:, :H, :] ** 2.2 ) * rgbScale.unsqueeze(1).unsqueeze(2), dim=(1, 2)))
        #print(rgbScale.shape, rgbScale.unsqueeze(1).unsqueeze(2).shape)
        albedo      = th.clamp( (mat[:, :H, :] ** 2.2 ) * rgbScale.unsqueeze(1).unsqueeze(2), 0.0, 1.0) ** (1/2.2)
        #albedo      = th.clamp( (mat[:, :H, :] ** 2.2 ) , 0.0, 1.0) ** (1/2.2)
        print(th.mean(albedo, dim=(1, 2)))

        normal      = mat[:, H:2*H, :]

        roughOriScale = th.mean(mat[:, 2*H:, :])
        roughScale    = roughMeanScale.cuda() / th.maximum(roughOriScale, th.ones_like(roughOriScale) * 1e-8)
        rough         = th.clamp(mat[:, 2*H:, :] * roughScale, 0.0, 1.0)
        #rough         = th.clamp(mat[:, 2*H:, :], 0.0, 1.0)

        mat = th.cat([albedo, normal, rough], dim=1)   
        print(mat.shape)     
        newMats.append(mat)
    return newMats

def mat2ref(mat): # 3 x 3H x W , between [0, 1], tensor
    mat = mat.permute(1, 2, 0) # 3H x W x 3
    H = mat.shape[0] // 3
    albedo = mat[:H, :, :]
    normal = mat[H:2*H, :, :]
    rough  = mat[2*H:, :, :]
    # tex: H x W x (3+2+1)
    tex = th.cat((albedo, normal[:,:,0:2], rough[:,:,0].unsqueeze(2)), 2)
    specular = 0.04 * th.ones_like(albedo)
    tex = th.cat((tex, specular),2)
    tex = tex * 2 - 1
    return tex.permute(2,0,1).unsqueeze(0).cuda()
    #return tex.permute(2,0,1).unsqueeze(0)

def optimLatentRender(args, target_ref, renderObj, envmap, bbox, planeNormal, latent_init_from, noise_init_from='random', res=256, \
                epochW=10, epochN=10, lr=0.02, oriMeans=None):
    if epochW == 0:
        if epochN == 0:
            optim_strategy = 'L+N'
        else:
            optim_strategy = 'N'
    else:
        if epochN == 0:
            optim_strategy = 'L'
        else:
            optim_strategy = 'L|N'
    genObj = StyleGAN2Generator('svbrdf')
    # initialize noise space
    global_var.init_global_noise(256, noise_init_from)
    print('\nInitial noise vector from ', noise_init_from)
    # initialize latent space
    latent = initLatent(genObj, type='w', init_from=latent_init_from)
    latent = Variable(latent, requires_grad=True)
    print('\nInitial latent vector from ', latent_init_from, ',', 'w')
    print('Latent vector shape:', latent.shape)
    # GAN generation
    texture_pre = updateTextureFromLatent(genObj, type='w', latent=latent)
    print('\nInitial texture maps from latent vector')
    texture_init = texture_pre.clone().detach()
    print('Initialized texture maps shape: ', texture_pre.shape)
    
    #rendered_init = renderObj.eval(texture_init, envmap)
    lossObj = LossesPPL(args, None, None, target_ref, res, None, None, None, oriMeans=oriMeans)
    for epoch in range(args.epochs):
        if optim_strategy == 'L+N':
            optimizer = Adam([latent] + global_var.noises, lr=lr, betas=(0.9, 0.999))
            optim_strategy_this = 'LN'
            # print('@@@ optim both @@@')
        elif optim_strategy == 'L':
            optimizer = Adam([latent], lr=lr, betas=(0.9, 0.999))
            optim_strategy_this = 'L'
            # print('@@@ optim latent @@@')
        elif optim_strategy == 'N':
            optimizer = Adam(global_var.noises, lr=lr, betas=(0.9, 0.999))
            optim_strategy_this = 'N'
            # print('@@@ optim noise @@@')
        else:
            epoch_tmp = epoch % (epochW+epochN)
            if int(epoch_tmp / epochW) == 0:
                optimizer = Adam([latent], lr=lr, betas=(0.9, 0.999))
                optim_strategy_this = 'L'
                # print('@@@ optim latent @@@')
            else:
                optimizer = Adam(global_var.noises, lr=lr, betas=(0.9, 0.999))
                optim_strategy_this = 'N'
                # print('@@@ optim noise @@@')

        # compute loss
        loss, loss_list = lossObj.eval(texture_pre, envmap, optim_strategy_this, epoch, bbox, planeNormal)
        # update latent/textures
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # undate textures
        texture_pre = updateTextureFromLatent(genObj, type='w', latent=latent)

        # save output
        lossMin = 1000
        if (epoch+1) % 100 == 0 or epoch == 0:
            now = datetime.now(); print(now)
            lossCurr = loss_list[0] * 100 / args.loss_weight[0] + loss_list[1] * 0.001 / args.loss_weight[1]
            print('[%d/%d]: optimization ... loss: ' % (epoch+1, args.epochs), loss_list, lossCurr)
            # renderAndSave(texture_pre, res, im_size, light_pos, camera_pos, light, args.num_render,
            #     out_this_dir, out_this_tmp_dir, epoch+1)
            #png = saveTex(texture_pre, out_this_dir, None, epoch+1)
            if lossCurr < lossMin:
                lossMin = lossCurr
                texture_best = texture_pre
                latent_best = latent
                noise_best = global_var.noises
                genObj_best = genObj
                epochMin = epoch
                rendered_pre = renderObj.eval(texture_pre, envmap)
                rendered_best = rendered_pre

            # if lossMin < 0.5 or (lossCurr >= lossMin and (epoch - epochMin) > 1000 ):
            #     break
    return latent_best, noise_best, texture_best, rendered_best

def optimLatentRender2(args, target_refList, renderObjList, envmapList, bboxList, planeNormalList, latent_init_from, noise_init_from='random', res=256, \
                epochW=10, epochN=10, lr=0.02, oriMeans=None):
    if epochW == 0:
        if epochN == 0:
            optim_strategy = 'L+N'
        else:
            optim_strategy = 'N'
    else:
        if epochN == 0:
            optim_strategy = 'L'
        else:
            optim_strategy = 'L|N'
    genObj = StyleGAN2Generator('svbrdf')
    # initialize noise space
    global_var.init_global_noise(256, noise_init_from)
    print('\nInitial noise vector from ', noise_init_from)
    # initialize latent space
    latent = initLatent(genObj, type='w', init_from=latent_init_from)
    latent = Variable(latent, requires_grad=True)
    print('\nInitial latent vector from ', latent_init_from, ',', 'w')
    print('Latent vector shape:', latent.shape)
    # GAN generation
    texture_pre = updateTextureFromLatent(genObj, type='w', latent=latent)
    print('\nInitial texture maps from latent vector')
    texture_init = texture_pre.clone().detach()
    print('Initialized texture maps shape: ', texture_pre.shape)
    
    #rendered_init = renderObj.eval(texture_init, envmap)
    lossObj = LossesPPL2(args, None, None, target_refList, res, None, None, None, oriMeans=oriMeans)
    for epoch in range(args.epochs):
        if optim_strategy == 'L+N':
            optimizer = Adam([latent] + global_var.noises, lr=lr, betas=(0.9, 0.999))
            optim_strategy_this = 'LN'
            # print('@@@ optim both @@@')
        elif optim_strategy == 'L':
            optimizer = Adam([latent], lr=lr, betas=(0.9, 0.999))
            optim_strategy_this = 'L'
            # print('@@@ optim latent @@@')
        elif optim_strategy == 'N':
            optimizer = Adam(global_var.noises, lr=lr, betas=(0.9, 0.999))
            optim_strategy_this = 'N'
            # print('@@@ optim noise @@@')
        else:
            epoch_tmp = epoch % (epochW+epochN)
            if int(epoch_tmp / epochW) == 0:
                optimizer = Adam([latent], lr=lr, betas=(0.9, 0.999))
                optim_strategy_this = 'L'
                # print('@@@ optim latent @@@')
            else:
                optimizer = Adam(global_var.noises, lr=lr, betas=(0.9, 0.999))
                optim_strategy_this = 'N'
                # print('@@@ optim noise @@@')

        # compute loss
        loss, loss_list = lossObj.eval(texture_pre, envmapList, optim_strategy_this, epoch, bboxList, planeNormalList)
        # update latent/textures
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # undate textures
        texture_pre = updateTextureFromLatent(genObj, type='w', latent=latent)

        # save output
        lossMin = 1000
        if (epoch+1) % 100 == 0 or epoch == 0:
            now = datetime.now(); print(now)
            lossCurr = loss_list[0] * 100 / args.loss_weight[0] + loss_list[1] * 0.001 / args.loss_weight[1]
            print('[%d/%d]: optimization ... loss: ' % (epoch+1, args.epochs), loss_list, lossCurr)
            # renderAndSave(texture_pre, res, im_size, light_pos, camera_pos, light, args.num_render,
            #     out_this_dir, out_this_tmp_dir, epoch+1)
            #png = saveTex(texture_pre, out_this_dir, None, epoch+1)
            if lossCurr < lossMin:
                lossMin = lossCurr
                texture_best = texture_pre
                latent_best = latent
                noise_best = global_var.noises
                genObj_best = genObj
                epochMin = epoch
                rendered_bestList = []
                for idx, renderObj in enumerate(renderObjList):
                    rendered_pre = renderObj.eval(texture_pre, envmapList[idx])
                    rendered_best = rendered_pre
                    rendered_bestList.append(rendered_best)


            # if lossMin < 0.5 or (lossCurr >= lossMin and (epoch - epochMin) > 1000 ):
            #     break
    return latent_best, noise_best, texture_best, rendered_bestList

def optimLatent(args, textures_ref, latent_init_from, noise_init_from='random', \
                epochW=10, epochN=10, lr=0.02):
    if epochW == 0:
        if epochN == 0:
            optim_strategy = 'L+N'
        else:
            optim_strategy = 'N'
    else:
        if epochN == 0:
            optim_strategy = 'L'
        else:
            optim_strategy = 'L|N'
    genObj = StyleGAN2Generator('svbrdf')
    # initialize noise space
    global_var.init_global_noise(256, noise_init_from)
    print('\nInitial noise vector from ', noise_init_from)
    # initialize latent space
    latent = initLatent(genObj, type='w', init_from=latent_init_from)
    latent = Variable(latent, requires_grad=True)
    print('\nInitial latent vector from ', latent_init_from, ',', 'w')
    print('Latent vector shape:', latent.shape)
    # GAN generation
    texture_pre = updateTextureFromLatent(genObj, type='w', latent=latent)
    print('\nInitial texture maps from latent vector')
    texture_init = texture_pre.clone().detach()
    print('Initialized texture maps shape: ', texture_pre.shape)
    lossObj = Losses(args, texture_init, textures_ref, None, None, None, None, None)
    for epoch in range(args.epochs):
        if optim_strategy == 'L+N':
            optimizer = Adam([latent] + global_var.noises, lr=lr, betas=(0.9, 0.999))
            optim_strategy_this = 'LN'
            # print('@@@ optim both @@@')
        elif optim_strategy == 'L':
            optimizer = Adam([latent], lr=lr, betas=(0.9, 0.999))
            optim_strategy_this = 'L'
            # print('@@@ optim latent @@@')
        elif optim_strategy == 'N':
            optimizer = Adam(global_var.noises, lr=lr, betas=(0.9, 0.999))
            optim_strategy_this = 'N'
            # print('@@@ optim noise @@@')
        else:
            epoch_tmp = epoch % (epochW+epochN)
            if int(epoch_tmp / epochW) == 0:
                optimizer = Adam([latent], lr=lr, betas=(0.9, 0.999))
                optim_strategy_this = 'L'
                # print('@@@ optim latent @@@')
            else:
                optimizer = Adam(global_var.noises, lr=lr, betas=(0.9, 0.999))
                optim_strategy_this = 'N'
                # print('@@@ optim noise @@@')

        # compute loss
        loss, loss_list = lossObj.eval(texture_pre, None, optim_strategy_this, epoch)
        # update latent/textures
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # undate textures
        texture_pre = updateTextureFromLatent(genObj, type='w', latent=latent)

        # save output
        lossMin = 1000
        if (epoch+1) % 100 == 0 or epoch == 0:
            now = datetime.now(); print(now)
            lossCurr = loss_list[0] * 100 / args.loss_weight[0] + loss_list[1] * 0.001 / args.loss_weight[1]
            print('[%d/%d]: optimization ... loss: ' % (epoch+1, args.epochs), loss_list, lossCurr)
            # renderAndSave(texture_pre, res, im_size, light_pos, camera_pos, light, args.num_render,
            #     out_this_dir, out_this_tmp_dir, epoch+1)
            #png = saveTex(texture_pre, out_this_dir, None, epoch+1)
            if lossCurr < lossMin:
                lossMin = lossCurr
                texture_best = texture_pre
                latent_best = latent
                noise_best = global_var.noises
                genObj_best = genObj
                epochMin = epoch

            # if lossMin < 0.5 or (lossCurr >= lossMin and (epoch - epochMin) > 1000 ):
            #     break
    return latent_best, noise_best, texture_best

def optimLatentFromMat(args, mats):
    latent_init_from = 'data/pretrain/latent_avg_W_256.pt'
    noise_init_from  = 'random'
    styles = []
    noises = []
    for idx, mat in enumerate(mats):
        textures_ref = mat2ref(mat)
        #####
        print('\n########## Optimizing material [%d/%d] ##########\n' % (idx+1, len(mats)))
        latent_best, noise_best, texture_best = optimLatent(args, textures_ref, latent_init_from, noise_init_from)
        #####
        styles.append(latent_best)
        noises.append(noise_best)
    return styles, noises

def nnsearchLatent(styles, latentDir):
    matList = glob.glob(os.path.join(latentDir, 'Material*'))
    dist = [100000000] * len(styles)
    nnNames = [''] * len(styles)
    for mid, matName in tqdm(enumerate(matList)): # ~ 2 minutes for 96248 materials
        # if mid % 100 == 0:
        #     print('[%d/%d]' % (mid+1, len(matList)))
        latentPath = os.path.join(matName, 'optim_latent.pt')
        matStyle = th.load(latentPath)
        for idx, style in enumerate(styles):
            d = th.mean( (matStyle - style)**2 )
            if d < dist[idx]:
                nnNames[idx] = matName
                dist[idx] = d

    return nnNames

def getRescaledMeanMatFromID(matIds, matMeans, oriMatRoot, matG1IdDict, res=128):
    # Apply scaled to specific materials
    svbrdfList = os.listdir(oriMatRoot)
    albedos = []
    normals = []
    roughs = []
    mats = []
    for i, matId in enumerate(matIds):
        matName = matG1IdDict[matId+1]
        if matName in svbrdfList:  # is svbrdf
            rgbMeanScale, roughMeanScale = mat2MeanScales(matMeans[i] ** 2.2)
            print(i)
            print(rgbMeanScale)
            albedoFile = os.path.join(oriMatRoot, matName,
                                      'tiled', 'diffuse_tiled.png')
            albedo = Image.open(albedoFile).convert('RGB')
            albedo = albedo.resize((res, res), Image.ANTIALIAS)
            albedo = np.asarray(albedo) / 255.0
            rgbOriScale = np.mean(albedo ** 2.2, axis=(0, 1))
            rgbScale    = rgbMeanScale.numpy() / np.maximum(rgbOriScale, 1e-8)
            albedo = np.clip(
                (albedo ** 2.2) * rgbScale[np.newaxis, np.newaxis, :], 0.0, 1.0) ** (1/2.2)
            print(np.mean(albedo, axis=(0, 1)))
            # albedo = Image.fromarray(np.uint8(albedo * 255))
            normal = np.asarray(Image.open(
                albedoFile.replace('diffuse', 'normal') ).resize((res, res), Image.ANTIALIAS) ) / 255.0

            rough = Image.open(albedoFile.replace(
                'diffuse', 'rough')).convert('L').resize((res, res), Image.ANTIALIAS)
            rough = np.asarray(rough) / 255.0
            roughOriScale = np.mean(rough)
            roughScale    = roughMeanScale.numpy() / np.maximum(roughOriScale, 1e-8)
            rough = np.clip(rough * roughScale, 0.0, 1.0)
            rough = np.tile(rough[:,:,np.newaxis], (1, 1, 3))
            # rough = Image.fromarray(np.uint8(rough * 255))
            mat = np.concatenate([albedo, normal, rough], axis=0)
            mats.append(th.from_numpy(np.transpose(mat, [2, 0, 1])))
            print(th.from_numpy(np.transpose(mat, [2, 0, 1])).shape)
        else:  # is homogeneous brdf    
            mats.append(matMeans[i])  

    return mats

def getRescaledMeanMatFromIDTopK(matIdsTopK, matMeans, oriMatRoot, matG1IdDict, res=128):
    # Apply scaled to specific materials
    svbrdfList = os.listdir(oriMatRoot)
    albedos = []
    normals = []
    roughs = []
    mats = []
    nMat, k = matIdsTopK.shape
    for i in range(nMat):
        matIdTopK = matIdsTopK[i] # k ids
        #k = matIdTopK.size(0)
        errMin = 100000
        for kk in range(k): # search for best mat among top-k
            matId = matIdTopK[kk]
            matName = matG1IdDict[matId+1]
            rgbMeanScale, roughMeanScale = mat2MeanScales(matMeans[i] ** 2.2)
            if matName in svbrdfList:  # is svbrdf
                albedoFile = os.path.join(oriMatRoot, matName,
                                        'tiled', 'diffuse_tiled.png')
                albedo = Image.open(albedoFile).convert('RGB')
                albedo = albedo.resize((res, res), Image.ANTIALIAS)
                albedo = np.asarray(albedo) / 255.0
                rgbOriScale = np.mean(albedo ** 2.2, axis=(0, 1))
                rgbScale    = rgbMeanScale.numpy() / np.maximum(rgbOriScale, 1e-8)
                albedo = np.clip(
                    (albedo ** 2.2) * rgbScale[np.newaxis, np.newaxis, :], 0.0, 1.0) ** (1/2.2)
                # albedo = Image.fromarray(np.uint8(albedo * 255))
                normal = np.asarray(Image.open(
                    albedoFile.replace('diffuse', 'normal') ).resize((res, res), Image.ANTIALIAS) ) / 255.0

                rough = Image.open(albedoFile.replace(
                    'diffuse', 'rough')).convert('L').resize((res, res), Image.ANTIALIAS)
                rough = np.asarray(rough) / 255.0
                roughOriScale = np.mean(rough)
                roughScale    = roughMeanScale.numpy() / np.maximum(roughOriScale, 1e-8)
                rough = np.clip(rough * roughScale, 0.0, 1.0)
                rough = np.tile(rough[:,:,np.newaxis], (1, 1, 3))
                # rough = Image.fromarray(np.uint8(rough * 255))
                errCurr = np.sum( (rgbOriScale - rgbMeanScale.numpy())**2 ) + (roughOriScale - roughMeanScale.numpy()) ** 2
                if errCurr < errMin:
                    mat = np.concatenate([albedo, normal, rough], axis=0)
                    matBest = th.from_numpy(np.transpose(mat, [2, 0, 1]))
                    errMin = errCurr
            else:  # is homogeneous brdf 
                _, vals = matName.split('__')
                r, g, b, rough = vals.split('_')
                rgb = np.array([float(r), float(g), float(b)])
                #rgbScale    = rgbMeanScale.numpy() / np.maximum(rgb, 1e-8)
                #rgb = np.clip(rgb * rgbScale, 0.0, 1.0) ** (1/2.2)
                #albedo = np.tile(rgb, (res, res, 1))
                #normal = np.tile(np.array([0.5, 0.5, 1]), [res, res, 1])
                #roughScale    = roughMeanScale.numpy() / np.maximum(float(rough), 1e-8)
                #rough = np.clip(float(rough) * roughScale, 0.0, 1.0)
                #rough = np.tile(rough, (res, res, 3))     
                errCurr = np.sum( (rgb - rgbMeanScale.numpy())**2 ) + (float(rough) - roughMeanScale.numpy()) ** 2
                if errCurr < errMin:
                    matBest = matMeans[i]
                    errMin = errCurr 
        mats.append(matBest)

    return mats

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

def saveMatNN(nnNames, matPredDir, partIdList, partNameDict):
    matSavePathDict = {}
    os.system('mkdir -p %s' % matPredDir)
    for idx, partId in enumerate(partIdList):
         # 3 x H x 4W, between [0, 1]
        matPred = th.from_numpy(loadTex(os.path.join(nnNames[idx], 'texOri.png'))).permute(2, 0, 1)
        W = matPred.shape[2] // 4
        albedoPath = os.path.join(matPredDir, '%s_diffuse.png' % partId)
        vutils.save_image(matPred[:, :, :W], albedoPath)
        normalPath = os.path.join(matPredDir, '%s_normal.png' % partId)
        vutils.save_image(matPred[:, :, W:2*W], normalPath)
        roughPath = os.path.join(matPredDir, '%s_rough.png' % partId)
        vutils.save_image(matPred[:, :, 2*W:3*W], roughPath)
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
    print('New Xml stored at %s' % xmlFileNew)

def test(opt):

    now = datetime.now()
    print(now)

    if opt.irMode == 'mean':
        modeStr = 'ir'
    elif opt.irMode == 'nn':
        modeStr = 'irnn'
    elif opt.irMode == 'optimcrop' or opt.irMode == 'optimcropreg' or opt.irMode == 'optimcrop2view':
        modeStr = 'ir%s' % opt.irMode
        opt.embed_tex = False
    elif opt.irMode == 'cs' or opt.irMode == 'csk' or opt.irMode == 'w':
        if opt.experiment is None:
            opt.experiment = 'check_%s_cw%d_sw%d_bn%d' % (
                opt.irMode.replace('csk', 'cs'), opt.classWeight, opt.scaleWeight, opt.batchSize)
        modelSavePath = os.path.join(opt.experiment, 'models')
        # Initial Network
        if opt.irMode == 'cs' or opt.irMode == 'csk':
            modeStr = 'ir%s' % opt.irMode
            net = models.netCS().cuda()
            net = nn.DataParallel(net, device_ids=opt.deviceIds)
        elif opt.irMode == 'w':
            modeStr = 'irw'
            genObj = StyleGAN2Generator('svbrdf')
            net = models.netW().cuda()
            net = nn.DataParallel(net, device_ids=opt.deviceIds)
        print('Loading model from %s' % os.path.join(modelSavePath, 'net_epoch_{}.pth'.format(opt.epochId)) )
        net.load_state_dict(torch.load(os.path.join(modelSavePath, 'net_epoch_{}.pth'.format(opt.epochId)) ) )
        print('Model %s is loaded!' % os.path.join(modelSavePath, 'net_epoch_{}.pth'.format(opt.epochId)) )
    
    brdfDataset = dataLoader.EvalSceneLoader(opt.sceneRoot, modelListRoot=opt.modelListRoot, 
                 rgbMode = 'imscannet', mode = modeStr, maskMode = opt.maskMode, vId = opt.vId)
    brdfLoader = DataLoader(brdfDataset, batch_size=1,
                            num_workers=8, shuffle=True)

    partNameDict = {}
    partIdDictPath = os.path.join(opt.sceneRoot, 'partIdDict.txt') # Will be generated first time initialization of EvalSceneLoader
    with open(partIdDictPath, 'r') as f:
        for line in f.readlines():
            partName, partId = line.strip().split(' ')
            partNameDict[partId] = partName

    #predRoot = os.path.join(opt.irPredRoot, opt.sceneRoot)
    matG1IdDict = getG1IdDict(os.path.join(opt.dataRoot, 'matIdGlobal1.txt'))

    startTime = timeit.default_timer()
    for i, dataBatch in enumerate(brdfLoader):
        #print('1. time : {}'.format(timeit.default_timer() - startTime) )
        # Load the image from cpu to gpu
        imBatch = Variable(dataBatch['im'][0]).cuda() # nSegs x 3 x h x w
        matMBatch = Variable(dataBatch['matMask'][0]).cuda() # nSegs x 1 x h x w
        viewId = dataBatch['xmlOutPath'][0].split('_')[-1].split('.')[0] # string

        if opt.irMode == 'cs' or opt.irMode == 'csk' or opt.irMode == 'w':
            depthBatch = Variable(dataBatch['depth'][0]).cuda() # nSegs x 1 x h x w
            inputBatch = torch.cat([imBatch, depthBatch, matMBatch.to(torch.float32)], dim=1)
            output = net(inputBatch)

        #print(predRoot)
        matsMean = getMeanMatFromIRNet(matMBatch, opt.irPredRoot, viewId, res=256)

        # Save predicted material maps and 
        print('partIdBatch: ', dataBatch['partId'])
        partIdList = dataBatch['partId'] # list of nSegs partIds
        partIdList = [x[0] for x in partIdList]
        print('matSegBatch: ', dataBatch['matPredDir'])
        matPredDir  = dataBatch['matPredDir'][0] # string
        if opt.irMode == 'mean':
            matSavePathDict = saveMatPred(matsMean, matPredDir, partIdList, partNameDict)
        elif opt.irMode == 'nn':
            styles, noises = optimLatentFromMat(args, matsMean)
            nnNames = nnsearchLatent(styles, args.matDataRoot)
            matSavePathDict = saveMatNN(nnNames, matPredDir, partIdList, partNameDict)
        elif opt.irMode == 'cs':
            #mat2MeanScales(mat): # 3 x 3H x W --> r,g,b,rough values
            _, matLabelPred = torch.max(output['material'].data, 1)
            assert(matLabelPred.size(0) == len(matsMean))
            matsPred = getRescaledMeanMatFromID(
                matLabelPred.cpu().numpy(), matsMean, opt.matOriDataRoot, matG1IdDict, res=256)
            matSavePathDict = saveMatPred(matsPred, matPredDir, partIdList, partNameDict)
        elif opt.irMode == 'csk':
            _, matLabelPredK = output['material'].data.topk(opt.k, 1, True, True)
            matsPred = getRescaledMeanMatFromIDTopK(
                matLabelPredK.cpu().numpy(), matsMean, opt.matOriDataRoot, matG1IdDict, res=256)
            matSavePathDict = saveMatPred(matsPred, matPredDir, partIdList, partNameDict)
        elif opt.irMode == 'w':
            matsPred = []
            for b in range(imBatch.size(0)):
                out = output['material'].detach()[b].unsqueeze(0)
                mP = getMatFromStyle(genObj, out, 'w') # 1 x 3 x 3H x W, [0, 1]
                matsPred.append(mP)
            matsPred = torch.cat(matsPred, dim=0)
            assert(matsPred.size(0) == len(matsMean))
            matsPred = rescaleToMean(matsPred, matsMean)
            matSavePathDict = saveMatPred(matsPred, matPredDir, partIdList, partNameDict)
        elif opt.irMode == 'optimcrop' or opt.irMode == 'optimcropreg':
            meanReg = True if opt.irMode == 'optimcropreg' else False
            #matsPred = getOptimFromCrop(imBatch, matMBatch, opt.irPredRoot, viewId, res=256)
            matSavePathDict = {}
            for partId in [8, 7, 1, 6, 2]:
                matsPred, matsSavePath = getOptimFromCropDemo(imBatch, matMBatch, opt.irPredRoot, viewId, partId, res=256, meanReg=meanReg)
                matSavePathDict[partNameDict[str(partId)]] = matsSavePath
            #matSavePathDict = saveMatPred(matsPred, matPredDir, partIdList, partNameDict)
        elif opt.irMode == 'optimcrop2view':
            meanReg = False
            #matsPred = getOptimFromCrop(imBatch, matMBatch, opt.irPredRoot, viewId, res=256)
            for partId in [8]:
                matsPred, matsSavePath = getOptimFromCropDemo2(imBatch, matMBatch, opt.irPredRoot, ['14', '154'], partId, res=256, meanReg=meanReg)

        # Save new xml file
        xmlFileOri = os.path.join(opt.sceneRoot, 'main.xml')
        xmlFileNew = dataBatch['xmlOutPath'][0] # string
        os.system('cp %s %s' % (xmlFileOri, xmlFileNew))
        saveNewXml(xmlFileNew, matSavePathDict, opt.isFast)


if __name__ == '__main__':
    print('\n\n\n')

    parser = argparse.ArgumentParser(description='PyTorch Optimization -- GY')
    # The locationi of training set
    parser.add_argument('--dataRoot', default=None,
                        help='path to input images')
    parser.add_argument('--matOriDataRoot', default=None,
                        help='path to material database')
    parser.add_argument('--matDataRoot', default=None,
                        help='path to material database')
    parser.add_argument('--sceneRoot', default=None)
    parser.add_argument('--vId', type=str, default='*', help='view id, default is *, set for only eval specific view')
    parser.add_argument('--modelListRoot', default=None)
    parser.add_argument('--objBaseRoot', default=None)
    parser.add_argument('--irPredRoot', default=None)
    parser.add_argument('--irMode', type=str, default='mean')
    parser.add_argument('--k', type=int, default=5, help='for irMode==csk, search among top k')
    parser.add_argument('--experiment', default=None,
                        help='the path to store samples and models')
    parser.add_argument('--fromServer', action='store_true', help='models copy from server are stored under pretrain/')
    parser.add_argument('--isFast', action='store_true', help='if true, modify xml file to use independent sampling')
    # The basic training setting
    parser.add_argument('--maskMode', type=str, default='', help='nothing or mmap, for mapped mask')
    parser.add_argument('--imHeight', type=int, default=240)
    parser.add_argument('--imWidth', type=int, default=320)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    # The training weight for the pre-trained model
    parser.add_argument('--classWeight', type=float, default=1.0,
                        help='the weight for the diffuse component')
    parser.add_argument('--scaleWeight', type=float, default=1.0,
                        help='the weight for the diffuse component')
    parser.add_argument('--batchSize', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--epochId', type=int, default=25,
                        help='the epoch ID for testing')
    parser.add_argument('--deviceIds', type=int, nargs='+',
                        default=[0], help='the gpus used for training network')
    # Parameters for optimization
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--applyMask', action='store_true')
    parser.add_argument('--applyMask2', action='store_true', help='full albedo, masked others')
    parser.add_argument('--diffuseOnly', action='store_true')
    parser.add_argument('--vgg_weight_dir', required=True)
    parser.add_argument('--embed_tex', action='store_true')
    parser.add_argument('--jittering', action='store_true')
    parser.add_argument('--loss_weight', type=float, nargs='+')
    args = parser.parse_args()

    # # ours
    # # args.vgg_layer_weight_w  = [0.125, 0.125, 0.125, 0.125]
    # # args.vgg_layer_weight_n  = [0.125, 0.125, 0.125, 0.125]

    args.vgg_layer_weight_w  = [0.071, 0.071, 0.286, 0.572]
    args.vgg_layer_weight_n  = [0.421, 0.421, 0.105, 0.053]

    # # args.vgg_layer_weight_w  = [0.071, 0.071, 0.286, 0.572]
    # # args.vgg_layer_weight_n  = [0.071, 0.071, 0.286, 0.572]

    # # args.vgg_layer_weight_w  = [0.421, 0.421, 0.105, 0.053]
    # # args.vgg_layer_weight_n  = [0.421, 0.421, 0.105, 0.053]

    # # args.vgg_layer_weight_w  = [0.421, 0.421, 0.105, 0.053]
    # # args.vgg_layer_weight_n  = [0.071, 0.071, 0.286, 0.572]

    args.vgg_layer_weight_wn = [0.125, 0.125, 0.125, 0.125]

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

    #assert args.mode in ['cs', 'w', 'w+', 'w+n', 'ir']
    torch.multiprocessing.set_start_method('spawn')
    test(args)
