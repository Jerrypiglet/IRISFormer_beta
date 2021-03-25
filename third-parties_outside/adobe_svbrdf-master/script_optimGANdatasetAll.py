import os

root_dir = 'data/'
cp_dir  = root_dir + 'pretrain/'
vgg_dir = cp_dir + 'vgg_conv.pt'

N = 7
epochs = 1000
epochW = 10
epochN = 10
loss = [100, 0.001, -1, -1]
lr = 0.02

datasetRoot = '/siggraphasia20dataset/BRDFScaledDataset'
out_dir = datasetRoot
#out_dir = 'test_out/'

matList = os.listdir(datasetRoot)
for mat_fn in matList:

    #mat_fn = 'Material__acetate_cloud'
    #in_dir = '/home/yyyeh/Datasets/BRDFOriginDataset/Material__acetate_cloud/tiled'
    in_dir = os.path.join(datasetRoot, mat_fn)

    cmd = 'CUDA_VISIBLE_DEVICES=1 python src/optimGANdataset.py' \
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