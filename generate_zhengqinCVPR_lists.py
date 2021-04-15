from pathlib import Path
ori_lists_path = Path('train/data/openrooms/list_OR_V4full/list')
zq_lists_path = Path('/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/DatasetCreation')
target_lists_path = Path('train/data/openrooms/list_OR_V4full_zhengqinCVPR/list')

# for my_split, zq_split in zip(['train', 'val'], ['train', 'test']):
my_split_total = []
for my_split in ['train', 'val', 'test']:
    ori_list = ori_lists_path / ('%s.txt'%my_split)
    with open(str(ori_list)) as f:
        mylist = f.read().splitlines() 
    my_split_total += mylist
    
for my_split, zq_split in zip(['train', 'val'], ['train', 'test']):
    zq_list = zq_lists_path / ('%s.txt'%zq_split)
    with open(str(zq_list)) as f:
        zqlist = f.read().splitlines()
    zq_scenes = [x for x in zqlist if 'scene' in x]
    my_split_filtered = [x for x in my_split_total if x.split(' ')[0] in zq_scenes]
    # print(my_split_filtered[0])
    # print(zq_split, len(my_split_total), len(my_split_filtered), len(zq_scenes))

    target_list_file = target_lists_path / ('%s.txt'%my_split)
    with open(str(target_list_file), 'w') as text_file:
        for x in my_split_filtered:
            text_file.write('%s\n'%(x))
    print('Wrote %d entries to %s'%(len(my_split_filtered), target_list_file))


