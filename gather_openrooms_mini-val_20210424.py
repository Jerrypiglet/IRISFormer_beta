from pathlib import Path, PurePath
from tqdm import tqdm
import random
import os
import shutil  

# list_path = 'train/data/openrooms'
if_cluster = True

# scene_list = ['scene0017_02', 'scene0053_00', 'scene0088_01', 'scene0120_00', 'scene0157_01', 'scene0195_01', 'scene0231_02', 'scene0269_00', 'scene0304_00', 'scene0344_01', 'scene0377_02', 'scene0414_00', 'scene0449_02', 'scene0483_00', 'scene0524_01', 'scene0562_00', 'scene0593_01']
dest_path_mini = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_mini-val'
list_path = '/home/ruizhu/Documents/Projects/semanticInverse/train/data/openrooms/list_ORmini-val'
original_list_path = '/home/ruizhu/Documents/Projects/semanticInverse/train/data/openrooms/list_OR_V4full/list/val.txt'

dataset_path_dict = {}
if if_cluster:
    dataset_path_dict['train'] = '/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/DatasetCreation'
    dataset_path_dict['test'] = ''
else:
    dataset_path_dict['train'] = ''
    dataset_path_dict['test'] = ''

# dirs = ['main_xml', 'main_xml1',
#         'mainDiffLight_xml', 'mainDiffLight_xml1', 
#         'mainDiffMat_xml', 'mainDiffMat_xml1']

subset_to_prefix = {'im_RGB': 'im_', 'label_semseg': 'imsemLabel_'}

list_read = open(original_list_path).readlines()
list_read = [x.strip() for x in list_read]

list_read = list_read[:500]

# for split in ['train', 'val']:
frame_paths_all_dict = {'im_RGB': [], 'label_semseg': []}
for subset in ['im_RGB', 'label_semseg']:
    frame_paths = [x.split(' ')[2] for x in list_read]
    if subset == 'label_semseg':
        frame_paths = [x.replace('mainDiffLight', 'main').replace('mainDiffMat', 'main').replace('im_', 'imsemLabel_').replace('.hdr', '.npy') for x in frame_paths]
    frame_paths_all_dict[subset] = frame_paths
    print(subset, frame_paths_all_dict[subset][:5])

meta_splits = [x.split('/')[0] for x in frame_paths_all_dict['im_RGB']]
scene_names = [x.split('/')[1] for x in frame_paths_all_dict['im_RGB']]
frame_ids = [x.split('/')[2].split('.')[0].split('_')[1] for x in frame_paths_all_dict['im_RGB']]

label_names = [
    ['imnormal_%s.png', 'main'], 
    ['im_%s.hdr', 'ori'], 
    ['imenv_%s.hdr', 'ori'], 
    # ['light_%s', ''], 
    ['immatPartGlobal1_%s.npy', 'ori'], 
    ['immatPartGlobal1Ids_%s.npy', 'ori'], 
    ['immatPartGlobal2_%s.npy', 'ori'], 
    ['immatPartGlobal2Ids_%s.npy', 'ori'], 
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
    # ['immatPart_%s.dat', ''], 
    # ['imsgEnv_%s.h5', ''], 
]

for idx, (meta_split, scene_name, frame_id) in tqdm(enumerate(zip(meta_splits, scene_names, frame_ids))):
    # print(meta_split, scene_name, frame_id)
    # original_scene_path = Path(dataset_path_dict['train']) / meta_split / scene_name
    # label_paths = [PurePath(x) for x in Path(original_scene_path).iterdir()]
    # label_paths = [x for x in label_paths if str(x).split('/')[-1].split('.')[0].split('_')[1] == frame_id]
    # if idx == 168:
    #     print('x for x in label_paths)
    # original_scene_path = Path(dataset_path_dict['train']) / 'main_xml' / scene_name
    # label_paths = [PurePath(x) for x in Path(original_scene_path).iterdir()]
    # label_paths = [x for x in label_paths if str(x).split('/')[-1].split('.')[0].split('_')[1] == frame_id]
    # for x in label_paths:
    #     print(x)
    
    # original_scene_path_main = str(original_scene_path).replace('DiffMat', '').replace('DiffLight', '')
    # original_scene_path_mainDiffMat = str(original_scene_path).replace('DiffMat', '')
    # original_scene_path_mainDiffLight = str(original_scene_path).replace('DiffLight', '')

    # original_scene_path_dict = {'ori': Path(original_scene_path), 'main': Path(original_scene_path_main), 'DiffMat': Path(original_scene_path_mainDiffMat), 'DiffLight': Path(original_scene_path_mainDiffLight)}
    original_meta_split_dict = {'ori': meta_split, 'main': meta_split.replace('DiffMat', '').replace('DiffLight', ''), 'DiffMat': meta_split.replace('DiffMat', ''), 'DiffLight': meta_split.replace('DiffLight', '')}


    # ------ copy files
    for label_name, meta_choice in label_names:
        meta_split_src = original_meta_split_dict[meta_choice]

        label_path =  Path(dataset_path_dict['train']) / meta_split_src / scene_name / (label_name%frame_id)
        assert label_path.exists(), str(label_path)
        src_path = str(label_path)

        dest_scene_path = Path(dest_path_mini) / meta_split_src / scene_name
        dest_scene_path.mkdir(exist_ok=True, parents=True)
        dest_path = str(dest_scene_path / label_path.name)

        # if Path(dest_path).exists() == False:
        if os.path.isfile(src_path):
            result = shutil.copyfile(src_path, dest_path)
        else:
            assert os.path.isdir(src_path)
            result = shutil.copytree(src_path, dest_path)
        # if 'immask' in src_path:
        #     print(idx, 'Copied from %s to %s'%(src_path, dest_path))

for split in ['train', 'val']:
    output_list_path = Path(list_path) / Path('list')
    output_list_path.mkdir(parents=True, exist_ok=True)
    output_txt_file = output_list_path / Path('%s.txt'%split)

    with open(str(output_txt_file), 'w') as text_file:
        for path_cam0, path_label in zip(frame_paths_all_dict['im_RGB'], frame_paths_all_dict['label_semseg']):
            scene_name = path_cam0.split('/')[1]
            frame_id = path_cam0.split('/')[2].split('.')[0].split('_')[1]
            text_file.write('%s %s %s %s\n'%(scene_name, frame_id, path_cam0, path_label))
    print('Wrote %d entries to %s'%(len(frame_paths_all_dict['im_RGB']), output_txt_file))




# ------ copy files
# for dir_ in tqdm(dirs):
#     for scene in scene_list:
#         src_path = Path(dataset_path_dict['train']) / dir_ / scene
#         dest_path = Path(dest_path_mini) / dir_ / scene
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


