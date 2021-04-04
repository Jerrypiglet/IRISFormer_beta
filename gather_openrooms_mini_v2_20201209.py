from pathlib import Path, PurePath
from tqdm import tqdm
import random
import os
import shutil  

# list_path = 'train/data/openrooms'
if_cluster = True

scene_list = ['scene0017_02', 'scene0053_00', 'scene0088_01', 'scene0120_00', 'scene0157_01', 'scene0195_01', 'scene0231_02', 'scene0269_00', 'scene0304_00', 'scene0344_01', 'scene0377_02', 'scene0414_00', 'scene0449_02', 'scene0483_00', 'scene0524_01', 'scene0562_00', 'scene0593_01']
dest_path_mini = '/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_mini'
list_path = dest_path_mini

dataset_path_dict = {}
if if_cluster:
    dataset_path_dict['train'] = '/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/DatasetCreation'
    dataset_path_dict['test'] = ''
else:
    dataset_path_dict['train'] = ''
    dataset_path_dict['test'] = ''

dirs = ['main_xml', 'main_xml1',
        'mainDiffLight_xml', 'mainDiffLight_xml1', 
        'mainDiffMat_xml', 'mainDiffMat_xml1']
# dirs = ['mainDiffMat_xml1']
# print(scene_list)

subset_to_prefix = {'im_RGB': 'im_', 'label_semseg': 'imsemLabel_'}
# subsample_ratio = 1.
# subsample_ratio_name_dict = {'1': '100k', '0.5': '50k', '0.3': '30k', '0.1': '10k'}
# assert subsample_ratio in subsample_ratio_name_dict.keys()

# for split in ['train', 'val', 'test']:
# for split in ['test']:
for dir_ in tqdm(dirs):
    for scene in scene_list:
        src_path = Path(dataset_path_dict['train']) / dir_ / scene
        dest_path = Path(dest_path_mini) / dir_ / scene
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        result = shutil.copytree(src_path, dest_path)
        print('Copied from %s to %s'%(src_path, dest_path))
        # print(os.listdir(path))


# for split in ['train', 'val']:
#     if split in ['train', 'val']:
#         dataset_path = dataset_path_dict['train']
#         scene_file = osp.join(dataset_path, 'train.txt')
#     else:
#         dataset_path = dataset_path_dict['test']
#         scene_file = osp.join(dataset_path, 'test.txt')
#     # with open(scene_file, 'r') as fIn:
#     #     scene_list = fIn.readlines() 
#     # scene_list = [x.strip() for x in scene_list]


#     frame_paths_all_dict = {'im_RGB': [], 'label_semseg': []}
#     scene_num = len(scene_list)
#     if split == 'train':
#         scene_list_split = scene_list[:int(scene_num*0.9)]
#     elif split == 'val':
#         scene_list_split = scene_list[-(scene_num - int(scene_num*0.9)):]
#     elif split == 'test':
#         scene_list_split = scene_list
#     # print('%d scenes for split %s'%(len(scene_list_split ), split))

#     scene_paths_split = []
#     for d in dirs:
#         scene_paths_split += [osp.join(dataset_path, d, x) for x in scene_list_split]
#     scene_paths_split = sorted(scene_paths_split)
#     scene_names_split = [PurePath(x).relative_to(dataset_path) for x in scene_paths_split]
#     print('Shape Num for split %s: %d' % (split, len(scene_names_split)), scene_names_split[:5] )

#     for scene_name in tqdm(scene_names_split):
#         scene_path = Path(dataset_path) / scene_name
#         frame_paths_dict = {'im_RGB': [], 'label_semseg': []}
#         for subset in ['im_RGB', 'label_semseg']:
#             subset_path = scene_path
#             frame_names = [PurePath(x).relative_to(subset_path) for x in Path(subset_path).iterdir()]
#             frame_paths = [str((subset_path / frame_name).relative_to(dataset_path)) for frame_name in frame_names]

#             frame_name_prefix = subset_to_prefix[subset]
#             frame_paths = [x for x in frame_paths if frame_name_prefix in str(x)]
#             frame_names = [x for x in frame_names if frame_name_prefix in str(x)]
#             frame_ids = [str(x).split('_')[1].split('.')[0] for x in frame_names]
#             # frame_paths.sort()
#             frame_paths = [x for _,x in sorted(zip(frame_names, frame_paths))]
#             # print('=====', subset, frame_name_prefix, frame_paths, frame_ids)
#             if subset == 'label_semseg':
#                 frame_paths = [x.replace('mainDiffLight', 'main').replace('mainDiffMat', 'main').replace('imsemLabel', 'imsemLabel') for x in frame_paths]
#             frame_paths_dict[subset] = frame_paths
#         # print(frame_paths_dict['im_RGB'])
#         # print(frame_paths_dict['label_semseg'])

#         # print(frame_paths_dict['im_RGB'])
#         # print(frame_paths_dict['label_semseg'])
#         if len(frame_paths_dict['im_RGB']) == 0:
#             print('No image files for scene %s. skipped.'%scene_name)
#             continue
        
#         if len(frame_paths_dict['im_RGB']) != len(frame_paths_dict['label_semseg']):
#             print('%d images != %d labels in scene %s'%(len(frame_paths_dict['im_RGB']), len(frame_paths_dict['label_semseg']), scene_name))
#             continue

#         for subset in ['im_RGB', 'label_semseg']:
#             frame_paths_all_dict[subset] += frame_paths_dict[subset]

#     for subset in ['im_RGB', 'label_semseg']:
#         random.seed(123456)
#         random.shuffle(frame_paths_all_dict[subset])
    
#     print(frame_paths_all_dict['im_RGB'][:5], frame_paths_all_dict['label_semseg'][:5])

#     if split == 'test':
#         # dataset_path = 'dataset/openrooms_test'
#         dataset_path = ''
#         print('Filtering out non-existent files...', dataset_path)
#         x_list, y_list = [], []
#         for x, y in tqdm(zip(frame_paths_all_dict['im_RGB'], frame_paths_all_dict['label_semseg'])):
#             print(osp.join(dataset_path, x), osp.join(dataset_path, y.replace('imsemLabel_', 'imsemLabel2_')))
#             if osp.isfile(osp.join(dataset_path, x)) and osp.isfile(osp.join(dataset_path, y)):
#                 x_list.append(x)
#                 y_list.append(y)
#         frame_paths_all_dict['im_RGB'], frame_paths_all_dict['label_semseg'] = x_list, y_list


#     # for subsample_ratio in list(subsample_ratio_name_dict.keys()):
#     #     subsample_ratio_float = float(subsample_ratio)
#     #     if subsample_ratio_float != 1.:
#     #         index_list = range(len(frame_paths_all_dict['im_RGB']))
#     #         sample_num = int(len(index_list) * subsample_ratio_float)
#     #         print(sample_num, len(index_list), subsample_ratio_float)
#     #         index_list_sample = random.sample(index_list, sample_num)
#     #         im_list = [frame_paths_all_dict['im_RGB'][i] for i in index_list_sample]
#     #         label_list = [frame_paths_all_dict['label_semseg'][i] for i in index_list_sample]
#     #     else:
#     im_list = frame_paths_all_dict['im_RGB']
#     label_list = frame_paths_all_dict['label_semseg']

    
    