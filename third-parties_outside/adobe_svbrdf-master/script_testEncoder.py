import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochId', type=str, default='24')
parser.add_argument('--classW', type=str, default='1', help='weight for classification loss')
parser.add_argument('--scaleW', type=str, default='1', help='weight for scale loss')
parser.add_argument('--batchSize', type=str, default='64')
parser.add_argument('--saveStep', type=str, default='500', help='weight for scale loss')
parser.add_argument('--mode', type=str, default='cs', help='type of material prediction')
parser.add_argument('--gpuId', type=str, default='0')
opt = parser.parse_args()

dataRoot = '/home/yyyeh/Datasets/OpenRoomTest/'
#dataRoot = '/siggraphasia20dataset/code/Routine/DatasetCreation/'
matOriDataRoot = '/home/yyyeh/Datasets/BRDFOriginDataset/'
#matOriDataRoot = '/siggraphasia20dataset/BRDFOriginDataset/'
matDataRoot = '/home/yyyeh/Datasets/BRDFScaledDatasetLatentW/'
#matDataRoot = '/eccv20dataset/BRDFScaledDatasetLatentW/'
mode = 'cs'  # cs, w, w+, w+n
if mode == 'w+n':
    matDataRoot = '/home/yyyeh/Datasets/BRDFScaledDatasetLatent/'
    #matDataRoot = '/eccv20dataset/BRDFScaledDatasetLatent/'

#cmd = 'CUDA_VISIBLE_DEVICES=%d python src/trainEncoder.py' % gpuId \
cmd = 'CUDA_VISIBLE_DEVICES=%s python3 src/testEncoder.py' % opt.gpuId \
    + ' --dataRoot ' + dataRoot \
    + ' --matDataRoot ' + matDataRoot \
    + ' --matOriDataRoot ' + matOriDataRoot \
    + ' --mode ' + mode \
    + ' --epochId ' + opt.epochId \
    + ' --saveStep ' + opt.saveStep \
    + ' --batchSize ' + opt.batchSize \
    + ' --cuda '

print(cmd)
os.system(cmd)
