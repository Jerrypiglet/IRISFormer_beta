import pickle5 as pickle
import os
if_render_other_modalities = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu')
parser.add_argument('--gpu_total', type=int, default=6, help='total num of gpus')
opt = parser.parse_args()

assert opt.gpu < opt.gpu_total

with open('/home/ruizhu/Downloads/tmp_cmds.pickle', 'rb') as f:
    cmds = pickle.load(f)

for cmd_idx, cmd in enumerate(cmds):
    gpu_idx = cmd_idx%6
    if gpu_idx!=opt.gpu:
        continue

    prefix = 'cd /home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_sequence_val_notSkipFrames && CUDA_VISIBLE_DEVICES=%d '%opt.gpu
    cmd = prefix + cmd
    # cmd = cmd.replace('cd /home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_sequence_val_notSkipFrames && CUDA_VISIBLE_DEVICES=3 ', prefix)
    # cmd = cmd.replace('/home/ruizhu/Documents/Projects/semanticInverse/renderImg_ScanNet_pose.py', 'renderImg_ScanNet_pose.py')
    print('----', cmd_idx, gpu_idx, cmd)
    # os.system(cmd)

    if if_render_other_modalities:
        for mode in [1, 2, 3, 4, 5, 6]: # L235 of /home/ruizhu/Documents/Projects/Total3DUnderstanding/OptixRenderer/src/optixRenderer/src/optixRenderer.cpp
            cmd_mod = cmd + ' --mode %d'%mode
            print('---->', cmd_idx, gpu_idx, cmd_mod)
            os.system(cmd_mod)
