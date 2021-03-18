import os
from tqdm import tqdm

root_dir = 'data/'
cp_dir  = root_dir + 'pretrain/'
vgg_dir = cp_dir + 'vgg_conv.pt'

N = 7
epochs = 1000
epochW = 10
epochN = 10
loss = [100, 0.001, -1, -1]
lr = 0.02

datasetRoot = '/home/yyyeh/Datasets/BRDFScaledDataset'
#out_dir = 'test_out/'

start = 47500
end   = 50000
gpuId = 7 
out_dir = datasetRoot + '_%d_%d_gpu%d' % (start, end, gpuId)

#matList = sorted(os.listdir(datasetRoot))
matListFile = '/home/yyyeh/Datasets/BRDFcode/matIdGlobal2.txt'
cnt = 0
#for mat_fn in matList:
with open(matListFile, 'r') as f:
    for line in tqdm(f.readlines()):
        if 'Material__' not in line:
            continue
        cnt += 1
        if cnt < start or cnt >= end:
            continue

        mat_fn, rs, gs, bs, roughs, _ = line.strip().split(' ')


        #mat_fn = 'Material__acetate_cloud'
        #in_dir = '/home/yyyeh/Datasets/BRDFOriginDataset/Material__acetate_cloud/tiled'
        in_dir = os.path.join(datasetRoot, mat_fn)

        cmd = 'CUDA_VISIBLE_DEVICES=%d python src/optimGANdataset.py' % gpuId \
            + ' --in_dir ' + in_dir \
            + ' --mat_fn ' + mat_fn \
            + ' --out_dir ' + out_dir \
            + ' --vgg_weight_dir ' + vgg_dir \
            + ' --num_render_used ' + str(N) \
            + ' --epochs ' + str(epochs) \
            + ' --sub_epochs ' + str(epochW) + ' ' + str(epochN) \
            + ' --loss_weight ' + str(loss[0]) + ' ' + str(loss[1])  + ' ' + str(loss[2]) + ' ' + str(loss[3]) \
            + ' --optim_latent' \
            + ' --lr ' + str(lr) \
            + ' --embed_tex' \
            + ' --gan_latent_init ' + cp_dir + 'latent_avg_W+_256.pt'
            # + ' --gan_latent_init ' + init_dir + mat_fn + '/optim_latent.pt' \
            # + ' --gan_noise_init ' + init_dir + mat_fn + '/optim_noise.pt' \
            # + ' --gan_latent_init ' + cp_dir + 'latent_const_W+_256.pt' \
            # + ' --gan_noise_init ' + cp_dir + 'latent_const_N_256.pt' \

            # + ' --jittering' \

        print(cmd)
        os.system(cmd)
