import sys
sys.path.insert(1, 'src/')
from util import *
import shutil
from optim import loadLightAndCamera
from script_gen_synthetic import render_image_pytorch

def preprocessing(mat, in_dir, tmp_dir):
    imres = 256

    tex = Image.fromarray(np.uint8(np.zeros((imres,imres*4,3))*255))
    im = Image.new('RGB', (imres*13, imres))

    for i in range(9):
        target = Image.open(os.path.join(in_dir, mat, '%02d.png' % i)).resize((imres, imres), Image.LANCZOS)
        im.paste(target, (imres*i,0))
    im.paste(tex, (imres*9,0))
    im.save(os.path.join(tmp_dir, mat+'.png'))

def postprocessing(mat, tmp_dir, out_dir, N):
    normal = Image.open(tmp_dir + str(N-1) + '/images/' + mat + '-outputs-0-.png')
    albedo = Image.open(tmp_dir + str(N-1) + '/images/' + mat + '-outputs-1-.png')
    rough  = Image.open(tmp_dir + str(N-1) + '/images/' + mat + '-outputs-2-.png')
    spec   = Image.open(tmp_dir + str(N-1) + '/images/' + mat + '-outputs-3-.png')

    albedo = gyApplyGammaPIL(albedo, 1/2.2)
    rough  = gyApplyGammaPIL(rough, 1/2.2)
    spec   = gyApplyGammaPIL(spec, 1/2.2)
    im = gyConcatPIL_h(gyConcatPIL_h(gyConcatPIL_h(albedo, normal), rough), spec)
    fn = os.path.join(out_dir, mat, 'tex.png')
    im.save(fn)
    # gyCreateThumbnail(fn, 128*4, 128)

def processing(tmp_dir, N):
    outputDir = tmp_dir
    inputDir = tmp_dir
    checkpoint = '../otherPaper/Deschaintre_egsr19/multiImage_code/checkpointTrained/'

    cmd = 'python ../otherPaper/Deschaintre_egsr19/multiImage_code/pixes2Material.py' \
        + ' --mode test' \
        + ' --output_dir ' + outputDir \
        + ' --input_dir ' + inputDir \
        + ' --batch_size 1' \
        + ' --input_size 256' \
        + ' --nbTargets ' + str(N-1) \
        + ' --useLog ' \
        + ' --includeDiffuse' \
        + ' --which_direction AtoB' \
        + ' --inputMode folder' \
        + ' --maxImages ' + str(N) \
        + ' --nbInputs 9 ' \
        + ' --feedMethod files' \
        + ' --useCoordConv' \
        + ' --checkpoint ' + checkpoint \
        + ' --fixImageNb'

    print(cmd)
    os.system(cmd)

def copyfiles(in_dir, out_dir):
    shutil.copyfile(in_dir+'camera_pos.txt', out_dir+'camera_pos.txt')
    shutil.copyfile(in_dir+'light_pos.txt', out_dir+'light_pos.txt')
    shutil.copyfile(in_dir+'image_size.txt', out_dir+'image_size.txt')
    shutil.copyfile(in_dir+'light_power.txt', out_dir+'light_power.txt')

def rendertexture(in_dir, out_dir):
    copyfiles(in_dir, out_dir)
    lp, cp, size, light = loadLightAndCamera(out_dir)
    render_image_pytorch(out_dir, out_dir, 256, size, lp, cp, light)


def eval(mat, in_dir, out_dir, tmp_dir, num_input):

    gyCreateFolder(out_dir)
    gyCreateFolder(out_dir + mat + '/')
    gyCreateFolder(tmp_dir)

    preprocessing(mat, in_dir, tmp_dir)
    processing(tmp_dir, num_input)
    postprocessing(mat, tmp_dir, out_dir, num_input)
    os.system('rm -r ' + tmp_dir)
    rendertexture(in_dir+mat+'/', out_dir+mat+'/')

if __name__ == '__main__':

    in_dir = 'data/in/'
    out_dir = 'data/out/egsr3/'
    tmp_dir = 'tmp/'

    num_input = 3

    mat_list = gyListNames(in_dir+'real_*')
    for j, mat in enumerate(mat_list):
        if True:
            print(mat)
            eval(mat, in_dir, out_dir, tmp_dir, num_input)
            # break
