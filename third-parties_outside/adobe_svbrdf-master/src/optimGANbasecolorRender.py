import global_var
from util import *
from render import *
from loss import *
from torch.optim import Adam
sys.path.insert(1, 'higan/models/')
from stylegan2_generator import StyleGAN2Generator
from torchvision import transforms

np.set_printoptions(precision=4, suppress=True)

# th.autograd.set_detect_anomaly(True)

def save_args(args, dir):
    with open(dir, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def loadLightAndCamera(in_dir):
    print('Load camera position from ', os.path.join(in_dir, 'camera_pos.txt'))
    camera_pos = np.loadtxt(os.path.join(in_dir, 'camera_pos.txt'), delimiter=',').astype(np.float32)

    print('Load light position from ', os.path.join(in_dir, 'light_pos.txt'))
    light_pos = np.loadtxt(os.path.join(in_dir, 'light_pos.txt'), delimiter=',').astype(np.float32)

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
        target = th.from_numpy(target).permute(2,0,1)
        return target

    rendered = th.zeros(num_render, 3, res, res)
    for i in range(num_render):
        rendered[i,:] = loadTargetToTensor(os.path.join(in_dir,'%02d.png' % i), res)
    rendered = rendered.cuda()

    texture_fn = os.path.join(in_dir, 'tex.png')
    if os.path.exists(texture_fn):
        textures, res0 = png2tex(texture_fn)
    else:
        textures = None

    return rendered, textures

def initTexture(init_from, res):
    if init_from == 'random':
        textures_tmp = th.rand(1,9,res,res)
        textures = textures_tmp.clone()
        textures[:,0:5,:,:] = textures_tmp[:,0:5,:,:] * 2 - 1
        textures[:,5,:,:] = textures_tmp[:,5,:,:] * 1.3 - 0.3
        textures[:,6:9,:,:] = textures_tmp[:,6:9,:,:] * 2 - 1
    else:
        textures, _ = png2tex(init_from)
        if res != textures.shape[-1]:
            print('The loaded initial texture has a wrong resolution!')
            exit()
    return textures

def initLatent(genObj, type, init_from):
    if init_from == 'random':
        if type == 'z':
            latent = th.randn(1,512).cuda()
        elif type == 'w':
            latent = th.randn(1,512).cuda()
            latent = genObj.net.mapping(latent)
        elif type == 'w+':
            latent = th.randn(1,512).cuda()
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
    textures_tmp[:,0:5,:,:] = textures[:,0:5,:,:].clamp(-1,1)
    textures_tmp[:,5,:,:] = textures[:,5,:,:].clamp(-0.3,1)
    textures_tmp[:,6:9,:,:] = textures[:,6:9,:,:].clamp(-1,1)

    return textures_tmp

def loadBasecolor(path):
    img = Image.open(path)
    img = img.convert('RGB')
    Tsfm = transforms.Compose([\
        transforms.Resize((256, 256)), \
        transforms.ToTensor()] )
    img = Tsfm(img) # 3 x 256 x 256
    return img.cuda()

def tex2pngConvert(tex, fn, isVertical=False):
    # isSpecular = False
    # if tex.size(1) == 9:
    #     isSpecular = True

    albedo, normal, rough, specular = tex2map(tex) # (x + 1) / 2) ** 2.2

    albedo = gyTensor2Array(albedo[0,:].permute(1,2,0))
    normal = gyTensor2Array((normal[0,:].permute(1,2,0)+1)/2)
    rough  = gyTensor2Array(rough[0,:].permute(1,2,0))
    specular = gyTensor2Array(specular[0,:].permute(1,2,0))

    # print(np.max(albedo + specular))
    basecolor = (albedo + specular) + np.sqrt((albedo + specular)**2 - 0.16 * albedo)
    # #metallic = np.zeros(basecolor[:,:,0].shape)
    # metallicList = []
    toLinGray = lambda rgbImg: 0.2126 * rgbImg[:,:,0] + 0.7152 * rgbImg[:,:,1] + 0.0722 * rgbImg[:,:,2] 
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
    specular2 = 0.04 * (1.0 - metallic[:, :, np.newaxis]) + basecolor * metallic[:, :, np.newaxis]

    albedo = gyArray2PIL(gyApplyGamma(albedo, 1/2.2))
    normal = gyArray2PIL(normal)
    rough  = gyArray2PIL(gyApplyGamma(rough, 1/2.2))
    specular = gyArray2PIL(gyApplyGamma(specular, 1/2.2))
    basecolor = gyArray2PIL(gyApplyGamma(np.clip(basecolor, 0.0, 1.0), 1/2.2))
    #metallic  = gyArray2PIL(gyApplyGamma(metallic, 1/2.2))
    metallic  = gyArray2PIL(metallic)
    albedo2 = gyArray2PIL(gyApplyGamma(albedo2, 1/2.2))
    specular2 = gyArray2PIL(gyApplyGamma(specular2, 1/2.2))


    if isVertical:
        png = gyConcatPIL_v(gyConcatPIL_v(albedo,specular), normal)
        png = gyConcatPIL_v(png, rough)
        png = gyConcatPIL_v(gyConcatPIL_v(png, basecolor), metallic)
        png = gyConcatPIL_v(gyConcatPIL_v(png, albedo2), specular2)
    else:
        png = gyConcatPIL_h(gyConcatPIL_h(albedo,specular), normal)
        png = gyConcatPIL_h(png, rough)
        png = gyConcatPIL_h(gyConcatPIL_h(png, basecolor), metallic)
        png = gyConcatPIL_h(gyConcatPIL_h(png, albedo2), specular2)

    if fn is not None:
        png.save(fn)
    return png

def saveTex(tex, save_dir, tmp_dir, idx):
    print('save_dir is %s' % save_dir)
    print('tmp_dir is %s' % tmp_dir)
    fn = os.path.join(save_dir,'tex%02d.png' % idx)
    #png = tex2pngConvert(tex, fn, isVertical=True)
    png = tex2png(tex, fn, isVertical=False)
    return png

def saveTexRef(tex_ref, save_dir):
    print('save_dir is %s' % save_dir)
    fn = os.path.join(save_dir,'tex_ref.png')
    #png = tex2pngConvert(tex, fn, isVertical=True)
    png = tex2png(tex_ref, fn, isVertical=False)
    return png

def renderAndSave(tex, res, size, lp, cp, li, num_render, save_dir, tmp_dir, epoch):
    fn = os.path.join(tmp_dir,'tex.png')
    fn2 = os.path.join(tmp_dir,'rendered.png')
    png = tex2png(tex, fn)
    # gyCreateThumbnail(fn,128*4,128)

    render_all = None
    for i in range(num_render):
        fn_this = save_dir + '/%02d.png' % i
        render_this = renderTex(fn, 256, size, lp[i,:], cp[i,:], li, fn_im=fn_this)
        # gyCreateThumbnail(fn_this)
        render_all = gyConcatPIL_h(render_all, render_this)
        png = gyConcatPIL_h(png, render_this)

    render_all.save(fn2)
    # gyCreateThumbnail(fn2, w=128*num_render, h=128)
    png.save(os.path.join(tmp_dir, 'epoch_%05d.jpg' % epoch))

def optim(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    th.manual_seed(args.seed)

    now = datetime.now(); print(now)

    if args.gpu > -1:
        th.cuda.set_device(args.gpu)

    mat = args.mat_fn
    res = args.im_res

    print('\n################ %s ###############' % mat)
    in_this_dir = os.path.join(args.in_dir, mat)
    out_this_dir = os.path.join(args.out_dir, mat)
    gyCreateFolder(out_this_dir)
    #out_this_tmp_dir = os.path.join(out_this_dir, 'tmp')
    out_this_tmp_dir = os.path.join(out_this_dir, args.target_mat)
    if args.diffuseOnly:
        out_this_tmp_dir += 'Diffuse'
    if args.applyMask:
        out_this_tmp_dir += 'Mask'
        if args.findInit:
            out_this_tmp_dir += 'Init'
    if args.applyMask2:
        out_this_tmp_dir += 'Mask'
    if args.sub_epochs[1] == 0:
        out_this_tmp_dir += '_L'
    if args.sub_epochs[0] == 0:
        out_this_tmp_dir += '_N'
    if args.alignPixMean:
        out_this_tmp_dir += '_apm'
    if args.alignPixStd:
        out_this_tmp_dir += '_aps'
    if args.alignVGG:
        out_this_tmp_dir += '_vgg'
    gyCreateFolder(out_this_tmp_dir)

    save_args(args, os.path.join(out_this_dir, 'args.txt'))
    print(args)

    light_pos, camera_pos, im_size, light = loadLightAndCamera(in_this_dir)
    light = light.astype('float32')
    print('\nlight_pos:\n', light_pos)
    print('\ncamera_pos:\n', camera_pos)
    print('\nim_size:\n', im_size)
    print('\nlight:\n', light)
    print('\n')

    #textures_ref = loadBasecolor(glob.glob(os.path.join(args.in_dir, '*baseColor.*') )[0] )
    textures_ref, _ = png2tex(args.target_path, args.mask_path, args.applyMask2)

    print(textures_ref.shape)
    # fn = os.path.join(out_this_dir, 'tex%s.png' % 'Ref')
    # png = Image.fromarray((textures_ref.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
    # png.save(fn)
    # print(glob.glob(os.path.join(args.in_dir, '*baseColor.*') )[0] )
    # rendered_ref, textures_ref = loadTarget(in_this_dir, res, args.num_render_used)
    # if args.embed_tex and textures_ref is None:
    #     print('No target maps find to embed!')
    #     exit()
    # print('\ntargets:\n', rendered_ref.shape)
    # if textures_ref is None:
    #     print('\nNo target maps!\n')
    # else:
    #     print('\ntarget maps:\n', textures_ref.shape)

    # initial textures
    if args.optim_latent:
        genObj = StyleGAN2Generator('svbrdf')
        # initialize noise space
        global_var.init_global_noise(res, args.gan_noise_init)
        print('\nInitial noise vector from ', args.gan_noise_init)
        # initialize latent space
        latent = initLatent(genObj, args.gan_latent_type, args.gan_latent_init)
        latent = Variable(latent, requires_grad=True)
        print('\nInitial latent vector from ', args.gan_latent_init, ',', args.gan_latent_type)
        print('Latent vector shape:', latent.shape)
        # GAN generation
        texture_pre = updateTextureFromLatent(genObj, args.gan_latent_type, latent)
        print('\nInitial texture maps from latent vector')
    else:
        texture_pre = initTexture(args.tex_init, res)
        texture_pre = Variable(texture_pre, requires_grad=True)
        print('\nInitial texture maps from ', args.tex_init)
    texture_init = texture_pre.clone().detach()
    print('Initialized texture maps shape: ', texture_pre.shape)

    # save initial texture and rendering
    renderAndSave(texture_pre, res, im_size, light_pos, camera_pos, light, args.num_render,
            out_this_dir, out_this_tmp_dir, 0)
    #png = saveTex(texture_pre, out_this_dir, out_this_tmp_dir, 0)
    _ = saveTexRef(textures_ref, out_this_tmp_dir)

    # initial loss
    loss_list_all = []
    loss_fn =  os.path.join(out_this_dir, 'loss.txt')
    #lossObj = Losses(args, texture_init, textures_ref, rendered_ref, res, im_size, light_pos, camera_pos)
    lossObj = Losses(args, texture_init, textures_ref, None, None, None, None, None)
    #lossObj = LossesMaps(args, textures_ref)

    if not args.optim_latent:
        optimizer = Adam([texture_pre], lr=args.lr, betas=(0.9, 0.999))
        optim_strategy_this = 'LN'

    # optimization
    min_loss = 1000
    for epoch in range(args.epochs):
        # update what to optimize
        if args.optim_latent:
            if args.optim_strategy == 'L+N':
                optimizer = Adam([latent] + global_var.noises, lr=args.lr, betas=(0.9, 0.999))
                optim_strategy_this = 'LN'
                # print('@@@ optim both @@@')
            elif args.optim_strategy == 'L':
                optimizer = Adam([latent], lr=args.lr, betas=(0.9, 0.999))
                optim_strategy_this = 'L'
                # print('@@@ optim latent @@@')
            elif args.optim_strategy == 'N':
                optimizer = Adam(global_var.noises, lr=args.lr, betas=(0.9, 0.999))
                optim_strategy_this = 'N'
                # print('@@@ optim noise @@@')
            else:
                epoch_tmp = epoch % (args.sub_epochs[0]+args.sub_epochs[1])
                if int(epoch_tmp / args.sub_epochs[0]) == 0:
                    optimizer = Adam([latent], lr=args.lr, betas=(0.9, 0.999))
                    optim_strategy_this = 'L'
                    # print('@@@ optim latent @@@')
                else:
                    optimizer = Adam(global_var.noises, lr=args.lr, betas=(0.9, 0.999))
                    optim_strategy_this = 'N'
                    # print('@@@ optim noise @@@')

        # compute loss
        # if args.diffuseOnly:
        #     if args.applyMask:
        #         loss, loss_list = lossObj.evalMasked(texture_pre, None, optim_strategy_this, epoch)
        #     else:
        #         loss, loss_list = lossObj.evalDiffuse(texture_pre, None, optim_strategy_this, epoch)
        # else:
        if args.applyMask:
            loss, loss_list = lossObj.evalMasked(texture_pre, None, optim_strategy_this, epoch)
        elif args.applyMask2:
            loss, loss_list = lossObj.evalMasked2(texture_pre, None, optim_strategy_this, epoch)
        else:
            loss, loss_list = lossObj.eval(texture_pre, None, optim_strategy_this, epoch)
        #loss, loss_list = lossObj.eval(texture_pre, light, optim_strategy_this, epoch)
        #loss, loss_list = lossObj.evalBasecolor(texture_pre, optim_strategy_this, epoch)
        loss_list_all.append(loss_list)
        np.savetxt(loss_fn, np.vstack(loss_list_all), fmt='%.4f', delimiter=',')

        # update latent/textures
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update textures
        if args.optim_latent:
            texture_pre = updateTextureFromLatent(genObj, args.gan_latent_type, latent)

        # save output
        if (epoch+1) % 100 == 0 or epoch == 0:
            now = datetime.now(); print(now)
            print('[%d/%d]: optimization ... loss: ' % (epoch+1, args.epochs), loss_list)
            renderAndSave(texture_pre, res, im_size, light_pos, camera_pos, light, args.num_render,
                out_this_dir, out_this_tmp_dir, epoch+1)
            #png = saveTex(texture_pre, out_this_dir, out_this_tmp_dir, epoch+1)

            plotAndSave(np.vstack(loss_list_all), os.path.join(out_this_tmp_dir, 'loss.png'))
            if args.optim_latent:
                if loss < min_loss:
                    print('latent vector saved at [%d/%d] with loss %0.4f' % (epoch+1, args.epochs, loss) )
                    th.save(latent, os.path.join(out_this_tmp_dir, 'optim_latent.pt'))
                    th.save(global_var.noises, os.path.join(out_this_tmp_dir, 'optim_noise.pt'))
                    min_loss = loss


    now = datetime.now(); print(now)
    print('Done!')


if __name__ == '__main__':
    print('\n\n\n')

    parser = argparse.ArgumentParser(description='PyTorch Optimization -- GY')
    parser.add_argument('--target_path', required=True)
    parser.add_argument('--target_mat', type=str, default='tmp')
    parser.add_argument('--diffuseOnly', action='store_true')
    parser.add_argument('--applyMask', action='store_true')
    parser.add_argument('--applyMask2', action='store_true', help='full albedo, masked others')
    parser.add_argument('--alignPixMean', action='store_true')
    parser.add_argument('--alignPixStd', action='store_true')
    parser.add_argument('--alignVGG', action='store_true')
    parser.add_argument('--mask_path', type=str, default=None)
    parser.add_argument('--findInit', action='store_true')
    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--mat_fn', type=str, default='')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--vgg_weight_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--im_res', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optim_latent', action='store_true')
    parser.add_argument('--tex_init', type=str, default='random')
    parser.add_argument('--num_render', type=int, default=9)
    parser.add_argument('--num_render_used', type=int, default=5)
    parser.add_argument('--gan_latent_type', type=str, default='w+')
    parser.add_argument('--gan_latent_init', type=str, default='random')
    parser.add_argument('--gan_noise_init', type=str, default='random')
    parser.add_argument('--sub_epochs', type=int, nargs='+')
    parser.add_argument('--loss_weight', type=float, nargs='+')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--embed_tex', action='store_true')
    parser.add_argument('--jittering', action='store_true')

    args = parser.parse_args()

    # ours
    # args.vgg_layer_weight_w  = [0.125, 0.125, 0.125, 0.125]
    # args.vgg_layer_weight_n  = [0.125, 0.125, 0.125, 0.125]

    args.vgg_layer_weight_w  = [0.071, 0.071, 0.286, 0.572]
    args.vgg_layer_weight_n  = [0.421, 0.421, 0.105, 0.053]

    # args.vgg_layer_weight_w  = [0.071, 0.071, 0.286, 0.572]
    # args.vgg_layer_weight_n  = [0.071, 0.071, 0.286, 0.572]

    # args.vgg_layer_weight_w  = [0.421, 0.421, 0.105, 0.053]
    # args.vgg_layer_weight_n  = [0.421, 0.421, 0.105, 0.053]

    # args.vgg_layer_weight_w  = [0.421, 0.421, 0.105, 0.053]
    # args.vgg_layer_weight_n  = [0.071, 0.071, 0.286, 0.572]

    args.vgg_layer_weight_wn = [0.125, 0.125, 0.125, 0.125]

    if args.sub_epochs[0] == 0:
        if args.sub_epochs[1] == 0:
            args.optim_strategy = 'L+N'
        else:
            args.optim_strategy = 'N'
    else:
        if args.sub_epochs[1] == 0:
            args.optim_strategy = 'L'
        else:
            args.optim_strategy = 'L|N'

    if args.seed:
        pass
    else:
        args.seed = random.randint(0, 2**31 - 1)

    args.epochs = max(args.sub_epochs[0] + args.sub_epochs[1], args.epochs)

    optim(args)