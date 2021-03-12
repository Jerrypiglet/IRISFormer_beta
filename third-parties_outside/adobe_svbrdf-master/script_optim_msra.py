import sys
sys.path.insert(1, 'src/')
from util import *
from optim import loadLightAndCamera
from script_gen_synthetic import render_image_pytorch

def preprocessing(mat, in_dir, init_dir, tmp_dir, N):
    im_res = 256

    tmp_target_dir = tmp_dir + 'target/'
    gyCreateFolder(tmp_target_dir)
    tmp_init_dir = tmp_dir + 'init/'
    gyCreateFolder(tmp_init_dir)
    tmp_light_dir = tmp_dir + 'wlv/'
    gyCreateFolder(tmp_light_dir)

    text_file = open(tmp_target_dir + 'files.txt', 'w')
    text_file.write(mat+'\n')
    text_file.close()

    for i in range(N):
        im = gyApplyGammaPIL(Image.open(os.path.join(in_dir, mat, '%02d.png' % i)), 1)
        im.resize((im_res,im_res), Image.LANCZOS).save(tmp_target_dir+mat+'_%02d.png' % i)

    tex = Image.open(init_dir)
    tex = gyPIL2Array(tex)
    albedo = gyArray2PIL(gyApplyGamma(tex[:,im_res*0:im_res*1,:], 1))
    normal = gyArray2PIL(tex[:,im_res*1:im_res*2,:])
    rough = gyArray2PIL(gyApplyGamma(tex[:,im_res*2:im_res*3,:], 2.2))
    spec = gyArray2PIL(gyApplyGamma(tex[:,im_res*3:im_res*4,:], 2.2))
    tex = gyConcatPIL_h(gyConcatPIL_h(gyConcatPIL_h(normal, albedo), rough), spec)
    fn = os.path.join(tmp_init_dir, mat + '.png')
    tex.save(fn)

    fn_light = os.path.join(in_dir, mat, 'light_pos.txt')
    lp = np.loadtxt(fn_light, delimiter=',').astype(np.float32)
    fn_camera = os.path.join(in_dir, mat, 'camera_pos.txt')
    cp = np.loadtxt(fn_camera, delimiter=',').astype(np.float32)
    fn_size = os.path.join(in_dir, mat, 'image_size.txt')
    image_size = np.loadtxt(fn_size, delimiter=',').astype(np.float32)
    lc_file = open(tmp_light_dir+mat+'.txt', 'w')
    lc_file.write('\n')
    for i in range(N):
        lc_file.write('%.4f,%.4f,%.4f %.4f,%.4f,%.4f\n' % (lp[i,0],lp[i,1],lp[i,2],cp[i,0],cp[i,1],cp[i,2]))
    lc_file.close()

    return image_size


def postprocessing(mat, tmp_dir, out_dir, num_input):

    tex = Image.open(tmp_dir + mat + '/output_1999.png')
    normal = tex.crop((0,0,256,256))
    albedo = tex.crop((256,0,256*2,256))
    rough = gyApplyGammaPIL(tex.crop((256*2,0,256*3,256)), 1/2.2)
    spec = gyApplyGammaPIL(tex.crop((256*3,0,256*4,256)), 1/2.2)
    tex = gyConcatPIL_h(gyConcatPIL_h(gyConcatPIL_h(albedo, normal), rough), spec)
    fn = os.path.join(out_dir, mat, 'tex.png')
    tex.save(fn)
    # gyCreateThumbnail(fn, 128*4, 128)


    tex = Image.open(tmp_dir + mat + '_refine/output_99.png')
    normal = tex.crop((0,0,256,256))
    albedo = tex.crop((256,0,256*2,256))
    rough = gyApplyGammaPIL(tex.crop((256*2,0,256*3,256)), 1/2.2)
    spec = gyApplyGammaPIL(tex.crop((256*3,0,256*4,256)), 1/2.2)
    tex = gyConcatPIL_h(gyConcatPIL_h(gyConcatPIL_h(albedo, normal), rough), spec)
    fn = os.path.join(out_dir[:-1]+'_refine/', mat, 'tex.png')
    tex.save(fn)
    # gyCreateThumbnail(fn, 128*4, 128)


def processing(tmp_dir, N, image_size):

    checkpoint = '../otherPaper/Gao_tog19/DeepInverseRendering/model/'

    cmd = 'CUDA_VISIBLE_DEVICES=0 python ../otherPaper/Gao_tog19/DeepInverseRendering/optimization/main.py' \
        + ' --N ' + str(N) \
        + ' --checkpoint ' + checkpoint \
        + ' --fileName files.txt' \
        + ' --dataDir ' + tmp_dir + 'target/' \
        + ' --logDir ' + tmp_dir \
        + ' --initDir ' + tmp_dir + 'init/' \
        + ' --network network_ae_fixBN' \
        + ' --init_method svbrdf' \
        + ' --input_type image' \
        + ' --wlv_type load' \
        + ' --wlvDir ' + tmp_dir + 'wlv/' \
        + ' --image_size ' + str(image_size) \
        + ' --light 1500' \
        + ' --max_steps 2000' \
        + ' --refine_max_steps 100' \
        + ' --lr 0.001'

    print(cmd)
    os.system(cmd)

def rendertexture(in_dir, out_dir):
    lp, cp, size, light = loadLightAndCamera(in_dir)
    render_image_pytorch(out_dir, out_dir, 256, size, lp, cp, light)

def eval(mat, in_dir, init_dir, out_dir, tmp_dir, num_input):

    gyCreateFolder(out_dir)
    gyCreateFolder(out_dir + mat + '/')
    gyCreateFolder(out_dir[:-1]+'_refine/')
    gyCreateFolder(out_dir[:-1]+'_refine/' + mat + '/')
    gyCreateFolder(tmp_dir)

    imsize = preprocessing(mat, in_dir, init_dir, tmp_dir, num_input)
    processing(tmp_dir, num_input, imsize)
    postprocessing(mat, tmp_dir, out_dir, num_input)
    os.system('rm -r ' + tmp_dir)
    rendertexture(in_dir+mat+'/', out_dir+mat+'/')
    rendertexture(in_dir+mat+'/', out_dir[:-1]+'_refine/'+mat+'/')

if __name__ == '__main__':

    for i in [5,9,13,17,21]:

        in_dir = 'data/in_tmp/'
        out_dir = 'data/out_tmp/msra%d_avg/' % i
        tmp_dir = 'tmp/'

        num_input = i

        mat_list = gyListNames(in_dir+'*')
        for j, mat in enumerate(mat_list):
            if mat == 'fake_018':
                print(mat)
                # init_dir = 'data/out/egsr1/' + mat + '/tex.png'
                # init_dir = 'data/pretrain/latent_const_256.png'
                init_dir = 'data/pretrain/latent_avg_256.png'
                eval(mat, in_dir, init_dir, out_dir, tmp_dir, num_input)
                # break
