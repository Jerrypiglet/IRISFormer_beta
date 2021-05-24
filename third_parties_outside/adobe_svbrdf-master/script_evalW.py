import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochId', type=str, default='23')
parser.add_argument('--classW', type=str, default='1', help='weight for classification loss')
parser.add_argument('--scaleW', type=str, default='1', help='weight for scale loss')
parser.add_argument('--batchSize', type=str, default='64')
#parser.add_argument('--saveStep', type=str, default='500', help='weight for scale loss')
parser.add_argument('--sceneId', type=str, default='0001_00')
parser.add_argument('--mode', type=str, default='w', help='type of material prediction')
parser.add_argument('--rgbMode', type=str, default='im', help='im or imscannet')
parser.add_argument('--maskMode', type=str, default='default', help='default or mmap, for mapped mask')
parser.add_argument('--gpuId', type=str, default='0')
parser.add_argument('--isFast', action='store_true')
opt = parser.parse_args()

machine = 'cluster'
if machine == 'cluster':
    dataRoot = '/siggraphasia20dataset/code/Routine/DatasetCreation/'
    matOriDataRoot = '/siggraphasia20dataset/BRDFOriginDataset/'
    matDataRoot = '/eccv20dataset/BRDFScaledDatasetLatentW/'
    if opt.mode == 'w+n':
        matDataRoot = '/eccv20dataset/BRDFScaledDatasetLatent/'
    modelListRoot = '/siggraphasia20dataset/models.txt'
    #sceneRoot = '/eccv20dataset/OpenRoomScanNetView/scene%s' % opt.sceneId
    sceneRoot = '/siggraphasia20dataset/code/Routine/DatasetCreation/OpenRoomScanNetView/scene%s' % opt.sceneId

elif machine == 'local':
    dataRoot = '/home/yyyeh/Datasets/OpenRoomTest/'
    matOriDataRoot = '/home/yyyeh/Datasets/BRDFOriginDataset/'
    matDataRoot = '/home/yyyeh/Datasets/BRDFScaledDatasetLatentW/'
    if opt.mode == 'w+n':
        matDataRoot = '/home/yyyeh/Datasets/BRDFScaledDatasetLatent/'
    modelListRoot = '/home/yyyeh/Datasets/BRDFcode/models.txt'

if opt.isFast:
    isFast = ' --isFast'
else:
    isFast = ''

#cmd = 'CUDA_VISIBLE_DEVICES=%d python src/trainEncoder.py' % gpuId \
cmd = 'CUDA_VISIBLE_DEVICES=%s python3 src/evalEncoder.py' % opt.gpuId \
    + ' --dataRoot ' + dataRoot \
    + ' --matOriDataRoot ' + matOriDataRoot \
    + ' --modelListRoot ' + modelListRoot \
    + ' --sceneRoot ' + sceneRoot \
    + ' --mode ' + opt.mode \
    + ' --rgbMode ' + opt.rgbMode \
    + ' --maskMode ' + opt.maskMode \
    + ' --epochId ' + opt.epochId \
    + ' --batchSize ' + opt.batchSize \
    + isFast \
    + ' --cuda '

print(cmd)
os.system(cmd)
