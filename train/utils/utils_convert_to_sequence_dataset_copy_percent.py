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
# DEST_path = Path('/newfoundland2/ruizhu/ORfull-seq-240x320-smaller-RE')
# DEST_percent_path = Path('/newfoundland2/ruizhu/ORfull-seq-240x320-smaller-RE-quarter')

LIST_path = Path('/viscompfs/users/ruizhu/semanticInverse/train/train/data/openrooms/list_OR_V4full/list')
DEST_path = Path('/ruidata/ORfull-seq-240x320-smaller-RE')
DEST_percent_path = Path('/ruidata/ORfull-seq-240x320-smaller-RE-quarter4')

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

valid_scene_list_quarter = []
valid_scene_list_all = []
for split in ['train', 'val']:
    valid_scene_split = []
    list_path = LIST_path / ('%s_scenes.txt'%split)
    scene_list_RAW = open(str(list_path)).readlines()
    for _ in scene_list_RAW:
        meta_split, scene_name = _.strip().split(' ')
        if 'scene' in scene_name:
            valid_scene_split.append([meta_split, scene_name])
            valid_scene_list_all.append([meta_split, scene_name])
    valid_scene_split = return_percent(valid_scene_split, PERCENT)
    valid_scene_list_quarter += valid_scene_split

print(valid_scene_list_quarter[:5])

print('Copying %d valid scenes (originally %d scenes)...'%(len(valid_scene_list_quarter), len(valid_scene_list_all)))

scene_list = []
for meta_split, scene_name in tqdm(valid_scene_list_quarter):
    for modality in modalities_convert:
        scene_list.append([modality, meta_split, scene_name])

def copy_scene(scene):
    modality, meta_split, scene_name = scene
    src_path = DEST_path / modality / meta_split / scene_name
    dest_path = DEST_percent_path / modality / meta_split / scene_name
    dest_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copytree(str(src_path), str(dest_path))
    
p = Pool(processes=32)
# p.map(copy_frame, valid_frame_list)
list(tqdm(p.imap_unordered(copy_scene, scene_list), total=len(scene_list)))

p.close()
p.join()
