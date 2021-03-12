import sys
sys.path.insert(1, 'src/')
from util import *
from dataset_tool import *
eps = 1e-6

N = 100  # -1 or N
in_dir = '/home/guoyu/Documents/3_svbrdf/otherPaper/Deschaintre_tog18/DeepMaterialsData/trainBlended/'
out_dir = '/media/guoyu/svbrdf_data/egsr_train/'

gyCreateFolder(out_dir+'tex/')
gyCreateFolder(out_dir+'npy/')
gyCreateFolder(out_dir+'tf/')
fn_list = gyListNames(in_dir + '*')


for j, fn in enumerate(fn_list):
    if j % 2 == 0:
        i = j/2
        if i < N or N < 0:
            print(fn)
            im = Image.open(in_dir + fn)
            # im.save(out_dir + fn)
            im = gyPIL2Array(im)

            normal = im[:,288*1:288*2,:]
            albedo = im[:,288*2:288*3,:]
            rough  = im[:,288*3:288*4,:]
            specular = im[:,288*4:288*5,:]

            normal_x = normal[:256, :256, 0]*2-1
            normal_y = normal[:256, :256, 1]*2-1
            normal_xy = np.clip(normal_x**2 + normal_y**2, a_min=0, a_max=1-eps)
            normal_z  = np.sqrt(1 - normal_xy)
            normal    = np.stack((normal_x, normal_y, normal_z), axis=2)
            normal   /= np.linalg.norm(normal, axis=2, keepdims=True)
            normal    = (normal + 1) / 2

            albedo    = gyApplyGamma(albedo[:256, :256, :], 1/2.2)
            specular  = gyApplyGamma(specular[:256, :256, :], 1/2.2)
            rough     = gyApplyGamma(rough[:256, :256, :], 1/2.2)

            tex = np.concatenate([albedo, normal, rough, specular], axis=1)
            gyArray2PIL(tex).save(out_dir + 'tex/%07d.png' % i)

            npy = np.concatenate([albedo, normal[:,:,0:2], np.expand_dims(rough[:,:,0], axis=-1), specular], axis=-1)
            np.save(out_dir + 'npy/%07d.npy' % i, (npy*255).astype(np.uint8))
        # exit()

create_siggraph_maps(out_dir+'tf/', out_dir+'npy/')

