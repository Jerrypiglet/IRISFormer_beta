import sys
sys.path.insert(1, 'src/')
from render import *
sys.path.insert(1, 'higan/models/')
from stylegan2_generator import StyleGAN2Generator
import global_var
from script_gen_synthetic import generateLightCameraPosition


merge = False
genTexFromStyleGAN2 = True
testGradOfStyleGAN2 = False
generateAvgLatent = False

if genTexFromStyleGAN2:
    out_dir = 'data/in_tmp/'

    latent_type = 'z'
    # latent_type = 'w'
    # latent_type = 'wp'

    global_var.init_global_noise(256, 'random')
    genObj = StyleGAN2Generator('svbrdf')
    print('Initialization Done!')
    N = 5
    for i in range(N**2):
        print(i)
        # z = genObj.easy_sample(1,latent_space_type='z')
        if latent_type == 'z':
            z = th.randn(1,512).cuda()
            w = genObj.net.mapping(z)
            wp = genObj.net.truncation(w)
        elif latent_type == 'w':
            w = th.randn(1,512).cuda()
            wp = genObj.net.truncation(w)
        elif latent_type == 'wp':
            wp = th.randn(1,14,512).cuda()

        print('create latent vevtor Done!')
        # results = genObj._synthesize(z, 'z')
        tex = genObj.net.synthesis(wp)
        gyCreateFolder(out_dir + 'tex_v')
        gyCreateFolder(out_dir + 'tex_h')
        png = tex2png(tex, None, isVertical=True)
        tex2png(tex, out_dir + 'tex_h/%02d.png' % i)
        print('generate image from latent space Done!')

        res = 256
        size = 20
        lp, cp = generateLightCameraPosition(20, 20, True, True)
        light = np.array([1500,1500,1500]).astype(np.float32)

        im_this = renderTex(out_dir + 'tex_h/%02d.png' % i, res, size, lp[0,:], cp[0,:], light, None)
        png = gyConcatPIL_v(png, im_this)
        png.save(out_dir + 'tex_v/%02d.png' % i)

    # for i in range(N):
    #     for j in range(N):
    #         k = i*N + j
    #         im_this = Image.open('tex_%02d.png' % k)
    #         if j > 0:
    #             im_h = gyConcatPIL_h(im_h, im_this)
    #         else:
    #             im_h = im_this
    #     if i > 0:
    #         im = gyConcatPIL_v(im, im_h)
    #     else:
    #         im = im_h

    # im.save('tex.png')

if merge:
    for i in range(16):
        tex = Image.open('data/in_tmp/tex_v/%02d.png' % i)
        render = Image.open('data/in_tmp/%02d/00.png' % i)
        gyConcatPIL_v(tex, render).save('data/in_tmp/%02d.png' % i)


if testGradOfStyleGAN2:

    genObj = StyleGAN2Generator('svbrdf')
    z = th.randn(1,512).cuda()
    z = Variable(z, requires_grad=True)
    w = genObj.net.mapping(z)
    wp = genObj.net.truncation(w)
    images = genObj.net.synthesis(wp)
    y = images.sum()
    if z.grad is not None:
        z.grad.data.zero_()
    y.backward()
    z_grad = gyTensor2Array(z.grad).reshape(512)

    delta = 1e-4
    z_grad_fd = np.zeros(512, dtype=np.float32)
    for i in range(512):
        z_this = z.clone()
        z_this[0,i] = z_this[0,i] + delta
        if z_this.grad is not None:
            z_this.grad.data.zero_()
        w_this = genObj.net.mapping(z_this)
        wp_this = genObj.net.truncation(w_this)
        images_this = genObj.net.synthesis(wp_this)
        y_this = images_this.sum()
        print('y     :', y)
        print('y_this:', y_this)
        z_grad_fd[i] = (gyTensor2Array(y_this)-gyTensor2Array(y))/delta
        print('z_grad   :', z_grad[i])
        print('z_grad_fd:', z_grad_fd[i])
    plt.plot(z_grad)
    plt.plot(z_grad_fd)
    plt.show()


if generateAvgLatent:
    genObj = StyleGAN2Generator('svbrdf')

    wp_all = th.zeros(5000,1,512, dtype=th.float32, device='cuda')
    for i in range(5000):
        z = th.randn(1,512).cuda()
        w = genObj.net.mapping(z)
        # wp = genObj.net.truncation(w)
        wp_all[i,:] = w
    wp_mean = wp_all.mean(0).detach().cpu()
    th.save(wp_mean, 'styleGAN_latent_avg_w_256.pt')
