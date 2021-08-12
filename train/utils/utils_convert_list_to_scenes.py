from pathlib import Path

# src_list_folder = Path('/home/ruizhu/Documents/Projects/semanticInverse/train/data/openrooms/list_ORmini/list/')
src_list_folder = Path('/home/ruizhu/Documents/Projects/semanticInverse/train/data/openrooms/list_OR_V4full/list/')
scene_name_dict = {}
frame_info_dict = {}
_list = []

for split in ['train', 'val']:
    scene_name_dict[split] = []
    frame_info_dict[split] = {}

    src_list_path = src_list_folder / ('%s.txt'%split)
    list_read = open(str(src_list_path)).readlines()
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        scene_name = line_split[0]
        meta_split = line_split[2].split('/')[0]
        frame_id = int(line_split[1])

        scene_key = '-'.join([meta_split, scene_name])
        scene_name_dict[split].append((meta_split, scene_name, scene_key))

        if scene_key in frame_info_dict[split]:
            frame_info_dict[split][scene_key].append(frame_id)
        else:
            frame_info_dict[split].update({scene_key: [frame_id]})

    scene_name_dict[split] = list(set(scene_name_dict[split]))

    for meta_split_scene_name in scene_name_dict[split]:
        assert meta_split_scene_name not in _list, 'scene already in another split!'
        _list.append(meta_split_scene_name)

    for _, _, scene_key in scene_name_dict[split]:
        frame_info_dict[split][scene_key] = list(set(frame_info_dict[split][scene_key]))
        frame_info_dict[split][scene_key].sort()
        frame_info_dict[split][scene_key] = [[scene_key, frame_id] for frame_id in frame_info_dict[split][scene_key]]

    frame_info_dict_to_list = [frame_info_dict[split][scene_key] for scene_key in frame_info_dict[split]]
    frame_info_dict_to_list = [item for sublist in frame_info_dict_to_list for item in sublist]

    dest_list_path = src_list_folder / ('%s_scenes.txt'%split)
    with open(dest_list_path, 'w') as f:
        for idx, x in enumerate(scene_name_dict[split]):
            f.write(' '.join([x[0], x[1]]))
            if idx != len(scene_name_dict[split])-1:
                f.write('\n')
    print('Written %d scenes to %s'%(len(scene_name_dict[split]), dest_list_path))

    dest_frame_info_list_path = src_list_folder / ('%s_scenes_frame_info.txt'%split)
    with open(dest_frame_info_list_path, 'w') as f:
        for idx, x in enumerate(frame_info_dict_to_list):
            f.write(' '.join([x[0], str(x[1])]))
            if idx != len(frame_info_dict_to_list)-1:
                f.write('\n')
    print('Written %d frames to %s'%(len(frame_info_dict_to_list), dest_frame_info_list_path))
