from os import pardir
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import h5py # https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html
'''
dtype ranges and sizes:
https://docs.oracle.com/cd/E19253-01/817-6223/chp-typeopexpr-2/index.html
https://clickhouse.tech/docs/en/sql-reference/data-types/int-uint/
'''

from multiprocessing import Pool
from utils_io import loadBinary
import argparse


def return_percent(list_in, percent=1.):
    len_list = len(list_in)
    return_len = max(1, int(np.floor(len_list*percent)))
    return list_in[:return_len]

PERCENT = 0.25

# LIST_path = Path('/home/ruizhu/Documents/Projects/semanticInverse/train/data/openrooms/list_OR_V4full/list')
# RAW_path = Path('/data/ruizhu/openrooms_mini')
# RAW_png_path = Path('/data/ruizhu/OR-pngs')
# DEST_path = Path('/home/ruizhu/Documents/data/OR-perFramePickles-mini-240x320-oneGiant')

LIST_path = Path('/viscompfs/users/ruizhu/semanticInverse/train/train/data/openrooms/list_OR_V4full/list')
RAW_path = Path('/siggraphasia20dataset/code/Routine/DatasetCreation/')
RAW_png_path = Path('/siggraphasia20dataset/pngs')
DEST_path = Path('/ruidata/ORfull-perFramePickles-240x320-oneGiant-quarter')


# RAW_path = Path('/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms')
# RAW_png_path = Path('/data/ruizhu/OR-pngs')
# DEST_path = Path('/newfoundland/ruizhu/ORfull-perFramePickles-240x320')

resize_HW = [240, 320] # set to [-1, -1] for not resizing!
dataset_if_save_space = True
if_resize = resize_HW!=[-1, -1]

# print(list(RAW_path.iterdir()))
# for x in tqdm():
parser = argparse.ArgumentParser()
parser.add_argument('--meta_split', type=str, default='NA', help='')
opt = parser.parse_args()

meta_splits_available = ['mainDiffLight_xml', 'mainDiffLight_xml1', 'mainDiffMat_xml', 'mainDiffMat_xml1', 'main_xml', 'main_xml1']
# meta_splits_available = ['mainDiffLight_xml']
modalities_convert = ['im_seg', 'depth', 'albedo']
# modalities_convert = ['im_seg']

if opt.meta_split != 'NA':
    assert opt.meta_split in meta_splits_available
    meta_splits = [opt.meta_split]
else:
    meta_splits = meta_splits_available

valid_frame_list_all = []

for split in ['train', 'val']:
    valid_frame_list_quarter = []
    list_path = LIST_path / ('%s.txt'%split)
    frame_list_RAW = open(str(list_path)).readlines()
    for _ in frame_list_RAW:
        scene_name = _.strip().split(' ')[0]
        meta_split = _.strip().split(' ')[2].split('/')[0]
        frame_id = int(_.strip().split(' ')[1])
        if 'scene' in scene_name:
            valid_frame_list_quarter.append([meta_split, scene_name, frame_id])

    valid_frame_list_quarter = return_percent(valid_frame_list_quarter, PERCENT)
    valid_frame_list_all += valid_frame_list_quarter


def process_meta_split(meta_split):

    meta_split_Path = RAW_path / meta_split
    scene_list = list(meta_split_Path.iterdir())

    DEST_path.mkdir(exist_ok=True, parents=True)
    dest_h5_file = DEST_path / ('%s.h5'%meta_split)
    hf = h5py.File(str(dest_h5_file), 'w')

    frame_count = 0
    # frame_info_list = []

    for src_scene_Path in scene_list:
        meta_split, scene_name = str(src_scene_Path).split('/')[-2:]

        if 'scene' not in scene_name:
            continue
            
        src_png_path = RAW_png_path / meta_split / scene_name

        # print(y.exists(), png_path.exists())
        if not src_png_path.exists():
            len_source_files = len(list(src_scene_Path.iterdir()))
            assert len_source_files == 0, 'Non-empty source path %s (%d files) with no png files at %s!'%(str(src_scene_Path), len_source_files, str(src_png_path))
            continue
            
        hdr_file_names = [_.name for _ in list(src_scene_Path.iterdir()) if ('im_' in _.name and '.hdr' in _.name)]
        if len(hdr_file_names)==0:
            continue

        sample_ids = [int(_.replace('im_', '').replace('.hdr', '')) for _ in hdr_file_names] # start with 1, ...
        sample_ids.sort()

        grp_scene = hf.create_group(scene_name)

        # print(meta_split, scene_name)

        for sample_id in sample_ids:
            if [meta_split, scene_name, sample_id] not in valid_frame_list_all:
                continue
            png_image_path = src_png_path / ('im_%d.png'%sample_id)
            assert png_image_path.exists()

            hdr_image_path = src_scene_Path / ('im_%d.hdr'%sample_id)

            grp_sample = grp_scene.create_group('%06d'%sample_id)

            if 'im_seg' in modalities_convert:
                im = Image.open(str(str(png_image_path)))
                if if_resize:
                    im = im.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
                im_uint8 = np.array(im)

                # im_uint8_list.append(im_uintgrp_sample
                grp_sample.create_dataset('im_uint8', data=im_uint8)

                seg_path = str(hdr_image_path).replace('im_', 'immask_').replace('hdr', 'png').replace('DiffMat', '')
                seg = Image.open(seg_path)
                if if_resize:
                    seg = seg.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
                seg_uint8 = np.asarray(seg, dtype=np.uint8)
                # seg_uint8_list.append(seg_uint8)
                grp_sample.create_dataset('seg_uint8', data=seg_uint8)

                semantics_path = str(hdr_image_path).replace('DiffMat', '').replace('DiffLight', '')
                mask_path = semantics_path.replace('im_', 'imcadmatobj_').replace('hdr', 'dat')
                mask_int32 = loadBinary(mask_path, channels = 3, dtype=np.int32, resize_HW=resize_HW).astype(np.uint16)
                assert np.amax(mask_int32) <= 65535 and np.amin(mask_int32) >= 0
                # print(mask_int32, np.amax(mask_int32))
                # print(mask_int32.shape)
                # .squeeze() # [h, w, 3]
                assert len(mask_int32.shape)==3
                # print(mask_int32.shape, mask_int32.dtype)
                # mask_int32_list.append(mask_int32)
                grp_sample.create_dataset('mask_int32', data=mask_int32)

            if 'albedo' in modalities_convert:
                albedo_path = str(hdr_image_path).replace('im_', 'imbaseColor_').replace('rgbe', 'png').replace('hdr', 'png')
                if dataset_if_save_space:
                    albedo_path = albedo_path.replace('DiffLight', '')
                albedo = Image.open(albedo_path)
                if if_resize:
                    albedo = albedo.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
                albedo_uint8 = np.asarray(albedo, dtype=np.uint8)
                # albedo_uint8_list.append(albedo_uint8)
                grp_sample.create_dataset('albedo_uint8', data=albedo_uint8)


            if 'depth' in modalities_convert:
                depth_path = str(hdr_image_path).replace('im_', 'imdepth_').replace('rgbe', 'dat').replace('hdr', 'dat')
                if dataset_if_save_space:
                    depth_path = depth_path.replace('DiffLight', '').replace('DiffMat', '')
                depth_float32 = loadBinary(depth_path, resize_HW=resize_HW)
                # depth_float32_list.append(depth_float32)
                grp_sample.create_dataset('depth_float32', data=depth_float32)

            frame_count += 1
            # print(frame_count)

        print(frame_count, str(src_png_path))
    hf.close()


    assert frame_count!= 0, str(src_png_path)

    return frame_count

import time
tic = time.time()

p = Pool(processes=6)
frame_info_list = p.map(process_meta_split, meta_splits)
p.close()
p.join()
        
print(sum(frame_info_list))
# frame_info_list_flat = [item for sublist in frame_info_list for item in sublist]
# print('%d frames converted.'%(sum(frame_info_list)))
# print('Took %.2f seconds'%(time.time()-tic))
