from genericpath import exists
from os import pardir
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from multiprocessing import Pool
from utils_io import loadBinary
import argparse
import shutil 


def return_percent(list_in, percent=1.):
    len_list = len(list_in)
    return_len = max(1, int(np.floor(len_list*percent)))
    return list_in[:return_len]

# RAW_path = Path('/data/ruizhu/openrooms_mini')
# RAW_png_path = Path('/data/ruizhu/OR-pngs')
# DEST_path = Path('/home/ruizhu/Documents/data/OR-seq-mini-240x320')

# DEST_path = Path('/ruidata/ORfull-seq-240x320-RE1smaller')

# LIST_path = Path('/home/ruizhu/Documents/Projects/semanticInverse/train/data/openrooms/list_OR_V4full/list')
# DEST_path = Path('/newfoundland2/ruizhu/ORfull-perFramePickles-240x320')
# DEST_percent_path = Path('/newfoundland2/ruizhu/ORfull-perFramePickles-240x320-quarter')

LIST_path = Path('/viscompfs/users/ruizhu/semanticInverse/train/train/data/openrooms/list_OR_V4full/list')
DEST_path = Path('/ruidata/ORfull-perFramePickles-240x320')
DEST_percent_path = Path('/ruidata/ORfull-perFramePickles-240x320-quarter')


PERCENT = 0.25

parser = argparse.ArgumentParser()
parser.add_argument('--meta_split', type=str, default='NA', help='')
opt = parser.parse_args()

meta_splits_available = ['mainDiffLight_xml', 'mainDiffLight_xml1', 'mainDiffMat_xml', 'mainDiffMat_xml1', 'main_xml', 'main_xml1']
# meta_splits_available = ['mainDiffLight_xml']
modalities_convert = ['im_png', 'depth', 'albedo']
# modalities_convert = ['im_seg']

if opt.meta_split != 'NA':
    assert opt.meta_split in meta_splits_available
    meta_splits = [opt.meta_split]
else:
    meta_splits = meta_splits_available

import time
tic = time.time()

valid_frame_list = []
for split in ['train', 'val']:
    list_path = LIST_path / ('%s.txt'%split)
    frame_list_RAW = open(str(list_path)).readlines()
    for _ in frame_list_RAW:
        scene_name = _.strip().split(' ')[0]
        meta_split = _.strip().split(' ')[2].split('/')[0]
        frame_id = int(_.strip().split(' ')[1])
        if 'scene' in scene_name:
            valid_frame_list.append([meta_split, scene_name, frame_id])

valid_frame_list = return_percent(valid_frame_list, PERCENT)
print('Copying %d valid scenes...'%len(valid_frame_list))

def copy_frame(frame_info):
    meta_split, scene_name, frame_id = frame_info[0], frame_info[1], frame_info[2]
    src_path = DEST_path / meta_split / scene_name / ('%06d.h5'%frame_id)
    dest_path = DEST_percent_path / meta_split / scene_name / ('%06d.h5'%frame_id)
    dest_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(str(src_path), str(dest_path))

p = Pool(processes=32)
# p.map(copy_frame, valid_frame_list)
list(tqdm(p.imap_unordered(copy_frame, valid_frame_list), total=len(valid_frame_list)))

p.close()
p.join()
