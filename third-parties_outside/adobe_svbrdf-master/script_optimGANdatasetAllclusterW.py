import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=None, help='start material id from 1 to 96249')
parser.add_argument('--end', type=int, default=None, help='end material id from 1 to 96249')
opt = parser.parse_args()

root_dir = 'data/'
cp_dir  = root_dir + 'pretrain/'
vgg_dir = cp_dir + 'vgg_conv.pt'

N = 7
epochs = 1000
epochW = 10
epochN = 10
loss = [100, 0.001, -1, -1]
lr = 0.02

datasetRoot = '/eccv20dataset/BRDFScaledDataset'
#datasetRoot = '/home/yyyeh/Datasets/BRDFScaledDataset'

start = opt.start
end   = opt.end
#out_dir = datasetRoot + '_%d_%d_gpu%d' % (start, end, gpuId)
#out_dir = datasetRoot + 'Latent'
out_dir = '/eccv20dataset/BRDFScaledDatasetLatentW'
progressFile = '/siggraphasia20dataset/code/Routine/DatasetCreation/material/genScaledLatentWProgress_%d_%d.txt' % (start, end)
existMatList = []
if os.path.exists(progressFile):
    with open(progressFile, 'r') as pf:
        for line in pf.readlines():
            mat = line.strip()
            existMatList.append(mat)

#matList = sorted(os.listdir(datasetRoot))
matListFile = '/siggraphasia20dataset/code/Routine/DatasetCreation/material/matIdGlobal2.txt'
#matListFile = '/home/yyyeh/Datasets/BRDFcode/matIdGlobal2.txt'
cnt = 0
#for mat_fn in matList:
with open(matListFile, 'r') as f:
    for line in tqdm(f.readlines()):
        if 'Material__' not in line:
            continue
        cnt += 1
        if cnt < start or cnt >= end:
            continue

        line2 = line.strip()
        if line2 in existMatList:
            print('%s exist! skip!' % line2)
            continue

        mat_fn, rs, gs, bs, roughs, _ = line.strip().split(' ')

        #mat_fn = 'Material__acetate_cloud'
        #in_dir = '/home/yyyeh/Datasets/BRDFOriginDataset/Material__acetate_cloud/tiled'
        in_dir = os.path.join(datasetRoot, mat_fn)
        out1 = os.path.join(out_dir, mat_fn, 'optim_latent.pt')
        out2 = os.path.join(out_dir, mat_fn, 'optim_noise.pt')
        if os.path.exists(out1) and os.path.exists(out2):
            print('%d: %s latent and noise exist! skip!' % (cnt, line2) )
            continue

        cmd = 'CUDA_VISIBLE_DEVICES=0 python3 src/optimGANdataset.py' \
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
            + ' --gan_latent_init ' + cp_dir + 'latent_avg_W_256.pt' \
            + ' --gan_latent_type w' 
            # + ' --gan_latent_init ' + init_dir + mat_fn + '/optim_latent.pt' \
            # + ' --gan_noise_init ' + init_dir + mat_fn + '/optim_noise.pt' \
            # + ' --gan_latent_init ' + cp_dir + 'latent_const_W+_256.pt' \
            # + ' --gan_noise_init ' + cp_dir + 'latent_const_N_256.pt' \

            # + ' --jittering' \

        print(cmd)
        os.system(cmd)

        with open(progressFile, 'a') as pf:
            pf.writelines('%s\n' % line2)
