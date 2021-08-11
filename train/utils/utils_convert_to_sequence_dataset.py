from os import pardir
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import h5py # https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html
from multiprocessing import Pool
from utils_io import loadBinary

RAW_path = Path('/data/ruizhu/openrooms_mini')
RAW_png_path = Path('/data/ruizhu/OR-pngs')
DEST_path = Path('/home/ruizhu/Documents/data/OR-seq-mini-240x320')

resize_HW = [240, 320] # set to [-1, -1] for not resizing!
dataset_if_save_space = True
if_resize = resize_HW!=[-1, -1]

# print(list(RAW_path.iterdir()))
# for x in tqdm():
meta_splits = ['mainDiffLight_xml', 'mainDiffLight_xml1', 'mainDiffMat_xml', 'mainDiffMat_xml1', 'main_xml', 'main_xml1']

def process_scene(src_scene_Path):
    frame_count = 0
    scene_name = src_scene_Path.name

    if 'scene' not in scene_name:
        return 0
        
    dest_path = DEST_path / meta_split / scene_name
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
    
    dest_path.mkdir(exist_ok=True, parents=True)

    im_uint8_list = []
    seg_uint8_list = []
    albedo_uint8_list = []
    depth_float32_list = []

    for sample_id in sample_ids:
        png_image_path = src_png_path / ('im_%d.png'%sample_id)
        assert png_image_path.exists()

        im = Image.open(str(str(png_image_path)))
        if if_resize:
            im = im.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
        im_uint8 = np.array(im)

        im_uint8_list.append(im_uint8)

        hdr_image_path = src_scene_Path / ('im_%d.hdr'%sample_id)
        seg_path = str(hdr_image_path).replace('im_', 'immask_').replace('hdr', 'png').replace('DiffMat', '')
        seg = Image.open(seg_path)
        if if_resize:
            seg = seg.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
        seg_uint8 = np.asarray(seg, dtype=np.uint8)
        seg_uint8_list.append(seg_uint8)

        albedo_path = str(hdr_image_path).replace('im_', 'imbaseColor_').replace('rgbe', 'png').replace('hdr', 'png')
        if dataset_if_save_space:
            albedo_path = albedo_path.replace('DiffLight', '')
        albedo = Image.open(albedo_path)
        if if_resize:
            albedo = albedo.resize([resize_HW[1], resize_HW[0]], Image.ANTIALIAS )
        albedo_uint8 = np.asarray(albedo, dtype=np.uint8)
        albedo_uint8_list.append(albedo_uint8)

        depth_path = str(hdr_image_path).replace('im_', 'imdepth_').replace('rgbe', 'dat').replace('hdr', 'dat')
        if dataset_if_save_space:
            depth_path = depth_path.replace('DiffLight', '').replace('DiffMat', '')
        depth_float32 = loadBinary(depth_path, resize_HW=resize_HW)
        depth_float32_list.append(depth_float32)

        frame_count += 1

    im_uint8_concat = np.stack(im_uint8_list) # [_, 240, 320, 3]
    seg_uint8_concat = np.stack(seg_uint8_list) # [_, 240, 320, 3]
    albedo_uint8_concat = np.stack(albedo_uint8_list) # [_, 240, 320, 3]
    depth_float32_concat = np.stack(depth_float32_list) # [_, 240, 320]
    # print(im_uint8_concat.shape)
    # print(seg_uint8_concat.shape)
    # print(albedo_uint8_concat.shape)
    # print(depth_float32_concat.shape)

    dest_h5_file = dest_path / 'im_png.h5'
    hf = h5py.File(str(dest_h5_file), 'w')
    hf.create_dataset('im_uint8', data=im_uint8_concat)
    hf.create_dataset('seg_uint8', data=seg_uint8_concat)
    hf.close()

    dest_h5_file = dest_path / 'albedo.h5'
    hf = h5py.File(str(dest_h5_file), 'w')
    hf.create_dataset('albedo_uint8', data=albedo_uint8_concat)
    hf.close()

    dest_h5_file = dest_path / 'depth.h5'
    hf = h5py.File(str(dest_h5_file), 'w')
    hf.create_dataset('depth_float32', data=depth_float32_concat)
    hf.close()

    # hf = h5py.File(str(dest_h5_file), 'r')
    # data_read = np.array(hf.get('data'))

    return frame_count

frame_counts_list = []
for meta_split in meta_splits:
    meta_split_Path = RAW_path / meta_split

    if 'main' not in meta_split:
        continue

    p = Pool(processes=16)
    frame_counts = p.map(process_scene, meta_split_Path.iterdir())
    frame_counts_list.append(frame_counts)
    print(meta_split, frame_counts)
    p.close()
    p.join()
        
frame_counts_list_flat = [item for sublist in frame_counts_list for item in sublist]
print('%d frames from %d scenes converted.'%(sum(frame_counts_list_flat), len(frame_counts_list_flat)))
