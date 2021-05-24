import os

root_dir = 'data/'
in_dir  = root_dir + 'in/'
#in_dir = '/mnt/ssd/tmp/yyeh/dataset/TestSetMaterials/AmericanElmWood'
#target_path = '/mnt/ssd/tmp/yyeh/dataset/TestSetMaterialsConverted/CandyCane.png'
#target_path = '/mnt/ssd/tmp/yyeh/dataset/TestSetMaterialsConverted/juese-temppbr-02.png'
#target_path = '/mnt/ssd/tmp/yyeh/dataset/TestSetMaterialsConverted/Adobe_asset_TikiTextured.png'
#target_path = '/mnt/ssd/tmp/yyeh/dataset/TestSetMaterialsConverted/combined_map_test.png'
#target_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/data/cow_mesh/cow_texture.png'
#target_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/data/cow_mesh/cow_texture_opt.png'
#target_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/data/Wood_Oak_Brown_Table/table.png'
#target_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/data/Wood_Oak_Brown_Table/table_opt.png'
#target_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/data/Teapot/teapot.png'
#target_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/data/Teapot/teapot_opt.png'
#target_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/data/Teapot/teapot_opt2.png'
target_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/data/Teapot/teapot/teapot_Mat_baseColor.png'
target_mat = target_path.split('/')[-1].split('.')[0]
#mask_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/result_cow/temp_multi_8_0_256/iter_500_loss_0.22_mask.png'
#mask_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/result_table1/temp_multi_8_0_256/iter_999_loss_0.20_mask.png'
#mask_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/result_teapot/temp_multi_8_0_256/iter_999_loss_0.22_mask.png'
mask_path = '/mnt/ilcompf0d0/user/yyeh/pytorch3d-scripts/data/Teapot/teapot_mask2.png'
out_dir = root_dir + 'out_tmp/ours7_avg/'
init_dir = root_dir + 'out/embed_egsr7/'
cp_dir  = root_dir + 'pretrain/'
vgg_dir = cp_dir + 'vgg_conv.pt'
gan_latent_init_dir = cp_dir + 'latent_avg_W+_256.pt'
#gan_latent_init_dir = '/mnt/ilcompf0d0/user/yyeh/material-hallucination/adobe_svbrdf-master/data/out_tmp/ours7_avg/real_plastic-red-carton/cow_texture_optDiffuseMaskInit/optim_latent.pt'
gpuId = 3

N = 7
epochs = 8000
epochW = 10
#epochW = 0
epochN = 10
#epochN = 0
#loss = [100, 0.001, -1, -1]
#loss = [100, 100, 100, -1]
loss = [100, 100, 100, 0.1]
lr = 0.02
mat_fn = 'real_plastic-red-carton'

cmd = 'CUDA_VISIBLE_DEVICES=' + str(gpuId) \
    + ' python src/optimGANbasecolorRender.py' \
    + ' --target_path ' + target_path \
    + ' --target_mat ' + target_mat \
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
    + ' --gan_latent_init ' + gan_latent_init_dir \
    + ' --applyMask2' \
    + ' --mask_path ' + mask_path \
    + ' --alignPixMean ' \
    + ' --alignPixStd ' \
    + ' --alignVGG' \
    # + ' --diffuseOnly' \
    # + ' --findInit ' \
    #+ ' --seed ' + str(131687955)
    # + ' --gan_latent_init ' + init_dir + mat_fn + '/optim_latent.pt' \
    # + ' --gan_noise_init ' + init_dir + mat_fn + '/optim_noise.pt' \
    # + ' --gan_latent_init ' + cp_dir + 'latent_const_W+_256.pt' \
    # + ' --gan_noise_init ' + cp_dir + 'latent_const_N_256.pt' \
    
    # + ' --jittering' \

print(cmd)
os.system(cmd)
