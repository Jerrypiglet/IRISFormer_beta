import os
import os.path as osp
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('--file', type=str, required=True )
#opt = parser.parse_args()
#print(opt )



os.system('bash /eccv20dataset/elvis/ECCV2021/setup.sh')

os.system('bash /eccv20dataset/elvis/ECCV2021/train.sh')

# cmd = '/siggraphasia20dataset/anaconda3/bin/python \
#         /siggraphasia20dataset/code/Routine/elvis/ECCV2021/%s' % (opt.file)
# print(cmd )
# os.system(cmd )
