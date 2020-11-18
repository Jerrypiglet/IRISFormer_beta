from pathlib import Path, PurePath
from tqdm import tqdm
import random
import os

list_path = 'train/data/openrooms'

dirs = ['main_xml', 'main_xml1',
        'mainDiffLight_xml', 'mainDiffLight_xml1', 
        'mainDiffMat_xml', 'mainDiffMat_xml1']
# print(scene_list)

subset_to_prefix = {'im_RGB': 'im_', 'label_semseg': 'imsemLabel_'}
# subsample_ratio = 1.
# subsample_ratio_name_dict = {'1': '100k', '0.5': '50k', '0.3': '30k', '0.1': '10k'}
# assert subsample_ratio in subsample_ratio_name_dict.keys()

# for split in ['train', 'val', 'test']:
for split in ['test']:
    if split in ['train', 'val']:
        dataset_path = 'dataset/openrooms'
        scene_file = os.path.join(dataset_path, 'train.txt')
    else:
        dataset_path = 'dataset/openrooms_test'
        scene_file = os.path.join(dataset_path, 'test.txt')
    with open(scene_file, 'r') as fIn:
        scene_list = fIn.readlines() 
    scene_list = [x.strip() for x in scene_list]


    frame_paths_all_dict = {'im_RGB': [], 'label_semseg': []}
    scene_num = len(scene_list)
    if split == 'train':
        scene_list_split = scene_list[:int(scene_num*0.95)]
    elif split == 'val':
        scene_list_split = scene_list[-(scene_num - int(scene_num*0.95)):]
    elif split == 'test':
        scene_list_split = scene_list
    # print('%d scenes for split %s'%(len(scene_list_split ), split))

    scene_paths_split = []
    for d in dirs:
        scene_paths_split += [os.path.join(dataset_path, d, x) for x in scene_list_split]
    scene_paths_split = sorted(scene_paths_split)
    scene_names_split = [PurePath(x).relative_to(dataset_path) for x in scene_paths_split]
    print('Shape Num for split %s: %d' % (split, len(scene_names_split)), scene_names_split[:5] )

    for scene_name in tqdm(scene_names_split):
        scene_path = Path(dataset_path) / scene_name
        frame_paths_dict = {'im_RGB': [], 'label_semseg': []}
        for subset in ['im_RGB', 'label_semseg']:
            subset_path = scene_path
            frame_names = [PurePath(x).relative_to(subset_path) for x in Path(subset_path).iterdir()]
            frame_paths = [str((subset_path / frame_name).relative_to(dataset_path)) for frame_name in frame_names]

            frame_name_prefix = subset_to_prefix[subset]
            frame_paths = [x for x in frame_paths if frame_name_prefix in str(x)]
            frame_names = [x for x in frame_names if frame_name_prefix in str(x)]
            frame_ids = [str(x).split('_')[1].split('.')[0] for x in frame_names]
            # frame_paths.sort()
            frame_paths = [x for _,x in sorted(zip(frame_names, frame_paths))]
            # print('=====', subset, frame_name_prefix, frame_paths, frame_ids)
            if subset == 'label_semseg':
                frame_paths = [x.replace('mainDiffLight', 'main').replace('mainDiffMat', 'main').replace('imsemLabel', 'imsemLabel') for x in frame_paths]
            frame_paths_dict[subset] = frame_paths
        # print(frame_paths_dict['im_RGB'])
        # print(frame_paths_dict['label_semseg'])

        # print(frame_paths_dict['im_RGB'])
        # print(frame_paths_dict['label_semseg'])
        if len(frame_paths_dict['im_RGB']) == 0:
            print('No image files for scene %s. skipped.'%scene_name)
            continue
        
        if len(frame_paths_dict['im_RGB']) != len(frame_paths_dict['label_semseg']):
            print('%d images != %d labels in scene %s'%(len(frame_paths_dict['im_RGB']), len(frame_paths_dict['label_semseg']), scene_name))
            continue

        for subset in ['im_RGB', 'label_semseg']:
            frame_paths_all_dict[subset] += frame_paths_dict[subset]

    for subset in ['im_RGB', 'label_semseg']:
        random.seed(123456)
        random.shuffle(frame_paths_all_dict[subset])
    
    print(frame_paths_all_dict['im_RGB'][:5], frame_paths_all_dict['label_semseg'][:5])

    if split == 'test':
        dataset_path = 'dataset/openrooms_test'
        print('Filtering out non-existent files...', dataset_path)
        x_list, y_list = [], []
        for x, y in tqdm(zip(frame_paths_all_dict['im_RGB'], frame_paths_all_dict['label_semseg'])):
            print(os.path.join(dataset_path, x), os.path.join(dataset_path, y.replace('imsemLabel_', 'imsemLabel2_')))
            if os.path.isfile(os.path.join(dataset_path, x)) and os.path.isfile(os.path.join(dataset_path, y)):
                x_list.append(x)
                y_list.append(y)
        frame_paths_all_dict['im_RGB'], frame_paths_all_dict['label_semseg'] = x_list, y_list


    # for subsample_ratio in list(subsample_ratio_name_dict.keys()):
    #     subsample_ratio_float = float(subsample_ratio)
    #     if subsample_ratio_float != 1.:
    #         index_list = range(len(frame_paths_all_dict['im_RGB']))
    #         sample_num = int(len(index_list) * subsample_ratio_float)
    #         print(sample_num, len(index_list), subsample_ratio_float)
    #         index_list_sample = random.sample(index_list, sample_num)
    #         im_list = [frame_paths_all_dict['im_RGB'][i] for i in index_list_sample]
    #         label_list = [frame_paths_all_dict['label_semseg'][i] for i in index_list_sample]
    #     else:
    im_list = frame_paths_all_dict['im_RGB']
    label_list = frame_paths_all_dict['label_semseg']

    
    output_list_path = Path(list_path) / Path('list')
    output_list_path.mkdir(parents=True, exist_ok=True)
    output_txt_file = output_list_path / Path('%s.txt'%split)
    # output_txt_file = output_txt_file.replace('.txt', '_%s.txt'%subsample_rario_name_dict[subsample_ratio])

    with open(str(output_txt_file), 'w') as text_file:
        for path_cam0, path_label in zip(frame_paths_all_dict['im_RGB'], frame_paths_all_dict['label_semseg']):
            text_file.write('%s %s\n'%(path_cam0, path_label))
    print('Wrote %d entries to %s'%(len(frame_paths_all_dict['im_RGB']), output_txt_file))
    