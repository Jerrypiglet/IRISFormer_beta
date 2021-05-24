import os
import glob

def gyListNames(in_dir):
    dir_list = sorted(glob.glob(in_dir))
    fn_list = []
    for dir in dir_list:
        fn_list.append(os.path.split(dir)[1])
    return fn_list

root_dir = 'data/'
in_dir  = root_dir + 'in_tmp/'
out_dir = root_dir + 'out_tmp/ours17_avg_refine/'
init_dir = root_dir + 'out_tmp/ours17_avg/'
cp_dir  = root_dir + 'pretrain/'
vgg_dir = cp_dir + 'vgg_conv.pt'

N = 17
epochs = 300
epochW = 10
epochN = 10
loss = [100, 0.01, -1, 50]
lr = 0.01

mat_list = gyListNames(in_dir + '*')

for id, mat in enumerate(mat_list):
    if mat=='fake_018':
        print(id, mat)

        cmd = 'python src/optim.py' \
            + ' --in_dir ' + in_dir \
            + ' --mat_fn ' + mat \
            + ' --out_dir ' + out_dir \
            + ' --vgg_weight_dir ' + vgg_dir \
            + ' --num_render 30 ' \
            + ' --num_render_used ' + str(N) \
            + ' --epochs ' + str(epochs) \
            + ' --sub_epochs ' + str(epochW) + ' ' + str(epochN) \
            + ' --loss_weight ' + str(loss[0]) + ' ' + str(loss[1]) + ' ' + str(loss[2]) + ' ' + str(loss[3])\
            + ' --lr ' + str(lr) \
            + ' --tex_init ' + init_dir + mat + '/tex.png' \

        print(cmd)
        os.system(cmd)
        # exit()
