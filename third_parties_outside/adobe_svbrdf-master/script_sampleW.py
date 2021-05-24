import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=str, default='8')
parser.add_argument('--gpuId', type=str, default='0')

opt = parser.parse_args()

machine = 'cluster'
if machine == 'cluster':
    matDataRoot = '/eccv20dataset/BRDFScaledDatasetLatentW/'
    sampleSavePath = '/eccv20dataset/adobe_svbrdf-master/sampleStyle'

elif machine == 'local':
    matDataRoot = '/home/yyyeh/Datasets/BRDFScaledDatasetLatentW/'

#cmd = 'CUDA_VISIBLE_DEVICES=%d python src/trainEncoder.py' % gpuId \
cmd = 'CUDA_VISIBLE_DEVICES=%s python3 src/sampleW.py' % opt.gpuId \
    + ' --matDataRoot ' + matDataRoot \
    + ' --batchSize ' + opt.batchSize \
    + ' --sampleSavePath ' + sampleSavePath \
    + ' --cuda '

print(cmd)
os.system(cmd)
