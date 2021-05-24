import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sceneId', type=str, default='0001_00')
parser.add_argument('--vId', type=str, default='*')
parser.add_argument('--maskMode', type=str, default='mmap', help='default or mmap, for mapped mask')
parser.add_argument('--irMode', type=str, default='optimcrop2view', help='mean, nn, optimcrop, optimcropreg, optimcrop2view, cs, csk, w')
parser.add_argument('--isFast', action='store_true')
opt = parser.parse_args()

if opt.irMode == 'cs' or opt.irMode == 'csk':
    epochId = 13
    batchSize = 64
    epochStr = ' --epochId ' + str(epochId)
    batchStr = ' --batchSize ' + str(batchSize)
elif opt.irMode == 'w':
    epochId = 23
    batchSize = 64
    epochStr = ' --epochId ' + str(epochId)
    batchStr = ' --batchSize ' + str(batchSize)
else:
    epochStr = ''
    batchStr = ''

machine = 'cluster'
if machine == 'cluster':
    dataRoot = '/siggraphasia20dataset/code/Routine/DatasetCreation/'
    matOriDataRoot = '/siggraphasia20dataset/BRDFOriginDataset/'
    matDataRoot = '/eccv20dataset/BRDFScaledDatasetLatentW/'
    # if opt.mode == 'w+n':
    #     matDataRoot = '/eccv20dataset/BRDFScaledDatasetLatent/'
    modelListRoot = '/siggraphasia20dataset/models.txt'
    #sceneRoot = '/eccv20dataset/OpenRoomScanNetView/scene%s' % opt.sceneId
    sceneRoot = '/siggraphasia20dataset/code/Routine/DatasetCreation/OpenRoomScanNetView/scene%s' % opt.sceneId
    irPredRoot = '/eccv20dataset/yyeh/InverseRenderNetPred/scene%s' % opt.sceneId
    gpuStr = ''

elif machine == 'local':
    dataRoot = '/home/yyyeh/Datasets/OpenRoomTest/'
    matOriDataRoot = '/home/yyyeh/Datasets/BRDFOriginDataset/'
    matDataRoot = '/home/yyyeh/Datasets/BRDFScaledDatasetLatentW/'
    # if opt.mode == 'w+n':
    #     matDataRoot = '/home/yyyeh/Datasets/BRDFScaledDatasetLatent/'
    modelListRoot = '/home/yyyeh/Datasets/BRDFcode/models.txt'
    gpuId = 1
    gpuStr = 'CUDA_VISIBLE_DEVICES=%d ' % gpuId

vgg_dir = 'data/pretrain/vgg_conv.pt'
loss = [100, 0.01, -1, -1]
epochs = 1000

if opt.isFast:
    isFast = ' --isFast'
else:
    isFast = ''

#cmd = 'CUDA_VISIBLE_DEVICES=%d python src/trainEncoder.py' % gpuId \
cmd = '%spython3 src/computeFromInvRenNetPred.py' % gpuStr \
    + ' --dataRoot ' + dataRoot \
    + ' --matOriDataRoot ' + matOriDataRoot \
    + ' --matDataRoot ' + matDataRoot \
    + ' --modelListRoot ' + modelListRoot \
    + ' --sceneRoot ' + sceneRoot \
    + ' --vId ' + opt.vId \
    + ' --irPredRoot ' + irPredRoot \
    + ' --irMode ' + opt.irMode \
    + ' --vgg_weight_dir ' + vgg_dir \
    + ' --embed_tex' \
    + ' --loss_weight ' + str(loss[0]) + ' ' + str(loss[1])  + ' ' + str(loss[2]) + ' ' + str(loss[3]) \
    + ' --epochs ' + str(epochs) \
    + epochStr \
    + ' --maskMode ' + opt.maskMode \
    + batchStr \
    + isFast \
    + ' --cuda '

print(cmd)
os.system(cmd)
