from pathlib import Path, PurePath
from tqdm import tqdm
import random
import os
import shutil
import numpy as np
import time
from multiprocessing import Pool

def return_percent(list_in, percent=1.):
    len_list = len(list_in)
    return_len = max(1, int(np.floor(len_list*percent)))
    return list_in[:return_len]

if_cluster = False

# DEST_PATH = Path('/ruidata/openrooms_raw_quarter')
DEST_PATH = Path('/ruidata/openrooms_raw_BRDF')
LIST_path = Path('/home/ruizhu/Documents/Projects/semanticInverse/train/data/openrooms/list_OR_V4full/list')
# PERCENT = 0.25
PERCENT = 1

dataset_path_dict = {}
if if_cluster:
    dataset_path_dict['train'] = '/siggraphasia20dataset/code/Routine/DatasetCreation'
    dataset_path_dict['test'] = ''
else:
    dataset_path_dict['train'] = '/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/DatasetCreation'
    dataset_path_dict['test'] = ''

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

    print('Total %d frames for [%s]...'%(len(valid_frame_list_quarter), split))
    if split == 'train':
        valid_frame_list_quarter = return_percent(valid_frame_list_quarter, PERCENT)
    print('Copying %d valid frames for [%s]...'%(len(valid_frame_list_quarter), split))
    print(valid_frame_list_quarter[:10])

    valid_frame_list_all += valid_frame_list_quarter

# assert ['main_xml', 'scene0570_00', 5] in valid_frame_list_all
# valid_scene_list_all = [(x[0], x[1]) for x in valid_frame_list_all]
# valid_scene_list_all = list(set(valid_scene_list_all))

# def create_scene(x):
#     meta_split, scene_name = x
#     dest_scene_path = Path(DEST_PATH) / meta_split / scene_name
#     dest_scene_path.mkdir(exist_ok=True, parents=True)

# tic = time.time()
# print('==== mkdir for %d scenes...'%len(valid_scene_list_all))
# # p = Pool(processes=16)
# # p.map(create_scene, valid_scene_list_all)
# # # list(tqdm(p.imap_unordered(copy_frame, valid_frame_list_all), total=len(valid_frame_list_all)))
# # p.close()
# # p.join()
# for scene_info_tuple in tqdm(valid_scene_list_all):
#     create_scene(scene_info_tuple)
# print('====> mkdir for %d scenes...DONE. Took %.2f seconds'%(len(valid_scene_list_all), time.time() - tic))


label_names = [
    ['imnormal_%s.png', 'main'], 
    ['im_%s.hdr', 'ori'], 
    ['imenv_%s.hdr', 'ori'], 
    # ['light_%s', ''], 
    # ['immatPartGlobal1_%s.npy', 'ori'], 
    # ['immatPartGlobal1Ids_%s.npy', 'ori'], 
    # ['immatPartGlobal2_%s.npy', 'ori'], 
    # ['immatPartGlobal2Ids_%s.npy', 'ori'], 
    # ['imsemLabel2_%s.npy', ''], 
    ['immask_%s.png', 'DiffMat'], 
    ['imdepth_%s.dat', 'main'], 
    # ['imsemLabel_%s.npy', ''], 
    ['imroughness_%s.png', 'DiffLight'], 
    # ['imshadingDirect_%s.rgbe', ''], 
    # ['imshading_%s.hdr', ''], 
    ['imbaseColor_%s.png', 'DiffLight'], 
    # ['imenvDirect_%s.hdr', ''], 
    ['imcadmatobj_%s.dat', 'main'], 
    # ['imsemLabel_%s.npy', 'main']
    # ['immatPart_%s.dat', ''], 
    # ['imsgEnv_%s.h5', 'ori'],
]

print('==== gnerating ori-dest tuple fro %d frames...'%len(valid_frame_list_all))

src_dest_path_list = []
dest_scene_path_list = []

for meta_split, scene_name, frame_id in tqdm(valid_frame_list_all):
    original_meta_split_dict = {'ori': meta_split, 'main': meta_split.replace('DiffMat', '').replace('DiffLight', ''), 'DiffMat': meta_split.replace('DiffMat', ''), 'DiffLight': meta_split.replace('DiffLight', '')}

    for label_name, meta_choice in label_names:
        meta_split_src = original_meta_split_dict[meta_choice]

        label_path =  Path(dataset_path_dict['train']) / meta_split_src / scene_name / (label_name%frame_id)
        # assert label_path.exists(), str(label_path)
        src_path = str(label_path)

        dest_scene_path = Path(DEST_PATH) / meta_split_src / scene_name
        dest_scene_path_list.append(dest_scene_path)
        # dest_scene_path.mkdir(exist_ok=True, parents=True)
        dest_path = str(dest_scene_path / label_path.name)

        src_dest_path_list.append((src_path, dest_path))
    
dest_scene_path_list = list(set(dest_scene_path_list))
tic = time.time()
print('==== mkdir for %d scenes...'%len(dest_scene_path_list))
for dest_scene_path in dest_scene_path_list:
    dest_scene_path.mkdir(exist_ok=True, parents=True)
print('====> mkdir for %d scenes...DONE. Took %.2f seconds'%(len(dest_scene_path_list), time.time() - tic))

def copy_frame(src_dest_path_tuple):
    src_path, dest_path = src_dest_path_tuple
    print(src_path, '-->', dest_path)

    # if os.path.isfile(src_path):
    result = shutil.copyfile(src_path, dest_path)
    # else:
    #     assert os.path.isdir(src_path)
    #     result = shutil.copytree(src_path, dest_path)

tic = time.time()
print('==== copying %d frames...'%len(valid_frame_list_all))
p = Pool(processes=32)
# p.map(copy_frame, src_dest_path_list)
list(tqdm(p.imap_unordered(copy_frame, src_dest_path_list), total=len(src_dest_path_list)))
p.close()
p.join()
print('==== copying %d frames...DONE. Took %.2f seconds'%(len(valid_frame_list_all), time.time() - tic))


# for idx, (meta_split, scene_name, frame_id) in tqdm(enumerate(valid_frame_list_all)):

# ------ copy files
# for dir_ in tqdm(dirs):
#     for scene in scene_list:
#         src_path = Path(dataset_path_dict['train']) / dir_ / scene
#         dest_path = Path(DEST_PATH) / dir_ / scene
#         dest_path.parent.mkdir(parents=True, exist_ok=True)
#         result = shutil.copytree(src_path, dest_path)
#         print('Copied from %s to %s'%(src_path, dest_path))

# ------ create lists
# for split in ['train', 'val']:
# # for split in ['val']:
#     frame_paths_all_dict = {'im_RGB': [], 'label_semseg': []}
#     dataset_path = dataset_path_dict['train']
#     scene_num = len(scene_list)
#     if split == 'train':
#         scene_list_split = scene_list[:int(scene_num*0.8)]
#     elif split == 'val':
#         scene_list_split = scene_list[-(scene_num - int(scene_num*0.8)):]
#     # elif split == 'test':
#     #     scene_list_split = scene_list

#     scene_paths_split = []
#     for d in dirs:
#         scene_paths_split += [os.path.join(dataset_path, d, x) for x in scene_list_split]
#     scene_paths_split = sorted(scene_paths_split)
#     scene_names_split = [PurePath(x).relative_to(dataset_path) for x in scene_paths_split]
#     print('Shape Num for split %s: %d' % (split, len(scene_names_split)), scene_names_split[:5] )

#     for scene_name in tqdm(scene_names_split):
#         scene_path = Path(dataset_path) / scene_name
#         frame_paths_dict = {'im_RGB': [], 'label_semseg': []}
#         for subset in ['im_RGB', 'label_semseg']:
#             subset_path = scene_path
#             frame_names = [PurePath(x).relative_to(subset_path) for x in Path(subset_path).iterdir()]
#             if len(frame_names) == 0:
#                 continue
#             frame_paths = [str((subset_path / frame_name).relative_to(dataset_path)) for frame_name in frame_names]

#             frame_name_prefix = subset_to_prefix[subset]
#             frame_paths = [x for x in frame_paths if frame_name_prefix in str(x)]
#             frame_names = [x for x in frame_names if frame_name_prefix in str(x)]
#             frame_ids = [str(x).split('_')[1].split('.')[0] for x in frame_names]
#             frame_paths = [x for _,x in sorted(zip(frame_names, frame_paths))]
#             if subset == 'label_semseg':
#                 frame_paths = [x.replace('mainDiffLight', 'main').replace('mainDiffMat', 'main').replace('imsemLabel', 'imsemLabel') for x in frame_paths]
#             frame_paths_dict[subset] = frame_paths

#         # if len(frame_paths_dict['im_RGB']) == 0:
#         #     print('No image files for scene %s. skipped.'%scene_name)
#         #     continue
        
#         if len(frame_paths_dict['im_RGB']) != len(frame_paths_dict['label_semseg']):
#             print('%d images != %d labels in scene %s'%(len(frame_paths_dict['im_RGB']), len(frame_paths_dict['label_semseg']), scene_name))
#             continue

#         for subset in ['im_RGB', 'label_semseg']:
#             frame_paths_all_dict[subset] += frame_paths_dict[subset]

#     # for subset in ['im_RGB', 'label_semseg']:
#     #     random.seed(123456)
#     #     random.shuffle(frame_paths_all_dict[subset])
    
#     print('===== %d frames for %s split!'%(len(frame_paths_all_dict['im_RGB']), split), frame_paths_all_dict['im_RGB'][:5], frame_paths_all_dict['label_semseg'][:5])


#     # output_list_path = Path(list_path) / Path('list')
#     # output_list_path.mkdir(parents=True, exist_ok=True)
#     # output_txt_file = output_list_path / Path('%s.txt'%split)
#     # # output_txt_file = output_txt_file.replace('.txt', '_%s.txt'%subsample_rario_name_dict[subsample_ratio])

#     # with open(str(output_txt_file), 'w') as text_file:
#     #     for path_cam0, path_label in zip(frame_paths_all_dict['im_RGB'], frame_paths_all_dict['label_semseg']):
#     #         scene_name = path_cam0.split('/')[1]
#     #         frame_id = path_cam0.split('/')[2].split('.')[0].split('_')[1]
#     #         text_file.write('%s %s %s %s\n'%(scene_name, frame_id, path_cam0, path_label))
#     # print('Wrote %d entries to %s'%(len(frame_paths_all_dict['im_RGB']), output_txt_file))


