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

# RAW_path = Path('/data/ruizhu/openrooms_mini')
# RAW_png_path = Path('/data/ruizhu/OR-pngs')
# DEST_path = Path('/home/ruizhu/Documents/data/OR-seq-mini-240x320')

RAW_path = Path('/siggraphasia20dataset/code/Routine/DatasetCreation/')
RAW_png_path = Path('/siggraphasia20dataset/pngs')
DEST_path = Path('/ruidata/ORfull-seq-240x320-RE1smaller')

# RAW_path = Path('/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms')
# RAW_png_path = Path('/data/ruizhu/OR-pngs')
# DEST_path = Path('/newfoundland/ruizhu/ORfull-seq-240x320')

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

def process_scene(src_scene_Path):
    frame_count = 0
    scene_name = src_scene_Path.name

    if 'scene' not in scene_name:
        return 0
    
    # if scene_name != 'scene0048_00':
    #     return 0
        
    src_png_path = RAW_png_path / meta_split / scene_name

    # print(y.exists(), png_path.exists())
    if not src_png_path.exists():
        len_source_files = len(list(src_scene_Path.iterdir()))
        assert len_source_files == 0, 'Non-empty source path %s (%d files) with no png files at %s!'%(str(src_scene_Path), len_source_files, str(src_png_path))
        return 0
        
    hdr_file_names = [_.name for _ in list(src_scene_Path.iterdir()) if ('im_' in _.name and '.hdr' in _.name)]
    if len(hdr_file_names)==0:
        return 0

    sample_ids = [int(_.replace('im_', '').replace('.hdr', '')) for _ in hdr_file_names] # start with 1, ...
    sample_ids.sort()
    
    sample_id_list = []
    im_uint8_list = []
    seg_uint8_list = []
    mask_int32_list = []
    albedo_uint8_list = []
    depth_float32_list = []

    for sample_id in sample_ids:
        png_image_path = src_png_path / ('im_%d.png'%sample_id)
        assert png_image_path.exists()

        sample_id_list.append(sample_id)

        hdr_image_path = src_scene_Path / ('im_%d.hdr'%sample_id)

        if 'im_seg' in modalities_convert:
            im = Image.open(str(str(png_image_path)))
            if if_resize:
                im = im.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
            im_uint8 = np.array(im)

            im_uint8_list.append(im_uint8)

            seg_path = str(hdr_image_path).replace('im_', 'immask_').replace('hdr', 'png').replace('DiffMat', '')
            seg = Image.open(seg_path)
            if if_resize:
                seg = seg.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
            seg_uint8 = np.asarray(seg, dtype=np.uint8)
            seg_uint8_list.append(seg_uint8)

            semantics_path = str(hdr_image_path).replace('DiffMat', '').replace('DiffLight', '')
            mask_path = semantics_path.replace('im_', 'imcadmatobj_').replace('hdr', 'dat')
            mask_int32 = loadBinary(mask_path, channels = 3, dtype=np.int32, resize_HW=resize_HW).astype(np.uint16)
            assert np.amax(mask_int32) <= 65535 and np.amin(mask_int32) >= 0
            # print(mask_int32, np.amax(mask_int32))
            # print(mask_int32.shape)
            # .squeeze() # [h, w, 3]
            assert len(mask_int32.shape)==3
            # print(mask_int32.shape, mask_int32.dtype)
            mask_int32_list.append(mask_int32)

        if 'albedo' in modalities_convert:
            albedo_path = str(hdr_image_path).replace('im_', 'imbaseColor_').replace('rgbe', 'png').replace('hdr', 'png')
            if dataset_if_save_space:
                albedo_path = albedo_path.replace('DiffLight', '')
            albedo = Image.open(albedo_path)
            if if_resize:
                albedo = albedo.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
            albedo_uint8 = np.asarray(albedo, dtype=np.uint8)
            albedo_uint8_list.append(albedo_uint8)

        if 'depth' in modalities_convert:
            depth_path = str(hdr_image_path).replace('im_', 'imdepth_').replace('rgbe', 'dat').replace('hdr', 'dat')
            if dataset_if_save_space:
                depth_path = depth_path.replace('DiffLight', '').replace('DiffMat', '')
            depth_float32 = loadBinary(depth_path, resize_HW=resize_HW)
            depth_float32_list.append(depth_float32)

        frame_count += 1

    if 'im_seg' in modalities_convert:
        im_uint8_concat = np.stack(im_uint8_list) # [_, 240, 320, 3]
        seg_uint8_concat = np.stack(seg_uint8_list) # [_, 240, 320, 3]
        mask_int32_concat = np.stack(mask_int32_list) # [_, 240, 320, 3]
        dest_path_img = DEST_path / 'im_png' / meta_split / scene_name
        dest_path_img.mkdir(exist_ok=True, parents=True)
        dest_h5_file = dest_path_img / 'im_png.h5'
        hf = h5py.File(str(dest_h5_file), 'w')
        hf.create_dataset('sample_id_list', data=sample_id_list)
        hf.create_dataset('im_uint8', data=im_uint8_concat)
        hf.create_dataset('seg_uint8', data=seg_uint8_concat)
        hf.create_dataset('mask_int32', data=mask_int32_concat)
        # print(type(sample_id_list[0]), im_uint8_concat.dtype, seg_uint8_concat.dtype, mask_int32_concat.dtype, )
        hf.close()
        assert im_uint8_concat.shape[0]==seg_uint8_concat.shape[0]==mask_int32_concat.shape[0]==frame_count

    if 'albedo' in modalities_convert:
        albedo_uint8_concat = np.stack(albedo_uint8_list) # [_, 240, 320, 3]
        dest_path_albedo = DEST_path / 'albedo' / meta_split / scene_name
        dest_path_albedo.mkdir(exist_ok=True, parents=True)
        dest_h5_file = dest_path_albedo / 'albedo.h5'
        hf = h5py.File(str(dest_h5_file), 'w')
        hf.create_dataset('albedo_uint8', data=albedo_uint8_concat)
        hf.close()
        assert albedo_uint8_concat.shape[0]==frame_count

    if 'depth' in modalities_convert:
        depth_float32_concat = np.stack(depth_float32_list) # [_, 240, 320]
        dest_path_depth = DEST_path / 'depth' / meta_split / scene_name
        dest_path_depth.mkdir(exist_ok=True, parents=True)
        dest_h5_file = dest_path_depth / 'depth.h5'
        hf = h5py.File(str(dest_h5_file), 'w')
        hf.create_dataset('depth_float32', data=depth_float32_concat)
        hf.close()
        assert depth_float32_concat.shape[0]==frame_count

    # print(im_uint8_concat.shape)
    # print(seg_uint8_concat.shape)
    # print(albedo_uint8_concat.shape)
    # print(depth_float32_concat.shape)
    



    # hf = h5py.File(str(dest_h5_file), 'r')
    # data_read = np.array(hf.get('data'))
    print(frame_count, str(src_png_path))
    assert frame_count!= 0, str(src_png_path)

    return frame_count

import time
tic = time.time()
frame_counts_list = []
for meta_split in meta_splits:
    meta_split_Path = RAW_path / meta_split

    if 'main' not in meta_split:
        continue

    p = Pool(processes=32)
    scene_list = list(meta_split_Path.iterdir())
    frame_counts = p.map(process_scene, scene_list)
    # frame_counts = list(tqdm(p.imap_unordered(process_scene, scene_list), total=len(scene_list)))

    frame_counts_list.append(frame_counts)
    # print(meta_split, frame_counts)
    p.close()
    p.join()
        
frame_counts_list_flat = [item for sublist in frame_counts_list for item in sublist]
print('%d frames from %d scenes converted.'%(sum(frame_counts_list_flat), len(frame_counts_list_flat)))
print('Took %.2f seconds'%(time.time()-tic))
