import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--classW', type=str, default='1', help='weight for classification loss')
parser.add_argument('--scaleW', type=str, default='1', help='weight for scale loss')
parser.add_argument('--saveStep', type=str, default='5000', help='weight for scale loss')
opt = parser.parse_args()

#dataRoot = '/home/yyyeh/Datasets/OpenRoom/'
dataRoot = '/siggraphasia20dataset/code/Routine/DatasetCreation/'
#matOriDataRoot = '/home/yyyeh/Datasets/BRDFOriginDataset/'
matOriDataRoot = '/siggraphasia20dataset/BRDFOriginDataset/'
#matDataRoot = '/home/yyyeh/Datasets/BRDFScaledDatasetLatentW/'
matDataRoot = '/eccv20dataset/BRDFScaledDatasetLatentW/'
mode = 'cs'  # cs, w, w+, w+n
if mode == 'w+n':
    #matDataRoot = '/home/yyyeh/Datasets/BRDFScaledDatasetLatent/'
    matDataRoot = '/eccv20dataset/BRDFScaledDatasetLatent/'

classW = opt.classW
scaleW = opt.scaleW

gpuId = 1
#cmd = 'CUDA_VISIBLE_DEVICES=%d python src/trainEncoder.py' % gpuId \
cmd = 'python3 src/trainEncoderTest.py' \
    + ' --dataRoot ' + dataRoot \
    + ' --matDataRoot ' + matDataRoot \
    + ' --matOriDataRoot ' + matOriDataRoot \
    + ' --mode ' + mode \
    + ' --classWeight ' + classW \
    + ' --scaleWeight ' + scaleW \
    + ' --saveStep ' + opt.saveStep \
    + ' --cuda '

print(cmd)
os.system(cmd)
