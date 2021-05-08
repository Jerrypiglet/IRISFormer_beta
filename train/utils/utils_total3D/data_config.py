"""
Created on May, 2019

@author: Yinyu Nie

Data configurations.

"""


class Relation_Config(object):
    def __init__(self):
        self.d_g = 64
        self.d_k = 64
        self.Nr = 16

num_samples_on_each_model = 5000
n_object_per_image_in_training = 8

import os
import numpy as np
# import pickle
import pickle5 as pickle

from pathlib import Path
from utils.utils_misc import red, basic_logger

NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

NYU37_TO_PIX3D_CLS_MAPPING = {0:0, 1:0, 2:0, 3:8, 4:1, 5:3, 6:5, 7:6, 8:8, 9:2, 10:2, 11:0, 12:0, 13:2, 14:4,
                              15:2, 16:2, 17:8, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:8, 25:8, 26:0, 27:0, 28:0,
                              29:8, 30:8, 31:0, 32:8, 33:0, 34:0, 35:0, 36:0, 37:8}

RECON_3D_CLS = [3,4,5,6,7,8,10,14,15,17,24,25,29,30,32]

# ------- OR42, OR45, OR46
OR4XCLASSES_dict = {'OR42': ['void', 
    'curtain', 'bike', 'washing_machine', 'table', 'dishwasher', 'bowl', 'bookshelf', 'sofa', 'speaker', 'trash_bin', 
    'piano', 'guitar', 'pillow', 'jar', 'bed', 'bottle', 'clock', 'chair', 'computer_keyboard', 'monitor', 
    'bathtub', 'stove', 'microwave', 'file_cabinet', 'flowerpot', 'cap', 'window', 'ceiling_lamp', 'telephone', 'printer', 
    'basket', 'faucet', 'bag', 'laptop', 'lamp', 'can', 'bench', 'door', 'cabinet', 'wall', 
    'floor', 'ceiling'],  # [0, ..., 42]

    'OR45': ['void',
    'curtain', 'bike', 'washing_machine', 'table', 'desk', 'pool_table', 'counter', 'dishwasher', 'bowl', 'bookshelf', 
    'sofa', 'speaker', 'trash_bin', 'piano', 'guitar', 'pillow', 'jar', 'bed', 'bottle', 'clock', 
    'chair', 'computer_keyboard', 'monitor', 'whiteboard', 'bathtub', 'stove', 'microwave', 'file_cabinet', 'flowerpot', 'cap', 
    'window', 'lamp', 'telephone', 'printer', 'basket', 'faucet', 'bag', 'laptop', 'can', 'bench', 
    'door', 'cabinet', 'wall', 'floor', 'ceiling'], 

    'OR46': ['void',
    'curtain', 'bike', 'washing_machine', 'table', 'desk', 'pool_table', 'counter', 'dishwasher', 'bowl', 'bookshelf', 
    'sofa', 'speaker', 'trash_bin', 'piano', 'guitar', 'pillow', 'jar', 'bed', 'bottle', 'clock', 
    'chair', 'computer_keyboard', 'monitor', 'whiteboard', 'bathtub', 'stove', 'microwave', 'file_cabinet', 'flowerpot', 'cap', 
    'window', 'lamp', 'telephone', 'printer', 'basket', 'faucet', 'bag', 'laptop', 'can', 'bench', 
    'door', 'cabinet', 'wall', 'floor', 'ceiling', 'ceiling_lamp']
}

OR4XCLASSES_not_detect_dict = {'OR42': ['void', 'curtain', 'pillow', 'cap', 'window', 'faucet', 'bag', 'bottle', 'can', 'door', 'wall', 
    'floor', 'ceiling', 'computer_keyboard', 'ceiling_lamp'], 
    'OR45': ['void', 'curtain', 'pillow', 'cap', 'window', 'faucet', 'bag', 'bottle', 'can', 'door', 'wall', 
    'floor', 'ceiling', 'computer_keyboard', 'lamp', 'whiteboard'], 
    'OR46': ['void', 'curtain', 'pillow', 'cap', 'window', 'faucet', 'bag', 'bottle', 'can', 'door', 'wall', 
    'floor', 'ceiling', 'computer_keyboard', 'lamp', 'ceiling_lamp', 'whiteboard'], 
}

OR4XCLASSES_not_detect_mapping_dict = {OR: {OR4XCLASSES_dict[OR].index(x): 0 for x in OR4XCLASSES_not_detect_dict[OR]} for OR in ['OR42', 'OR45', 'OR46']}
OR4XCLASSES_not_detect_mapping_ids_dict = {OR: list(OR4XCLASSES_not_detect_mapping_dict[OR].keys()) for OR in ['OR42', 'OR45', 'OR46']}

# OR42CLASSES_not_vis = OR42CLASSES_not_detect
# OR42CLASSES_not_vis = ['bike', 'washing_machine', 'table', 'dishwasher', 'bookshelf', 'sofa', 'trash_bin', 
#     'piano', 'bed', 'chair', 
#     'bathtub', 'stove', 'file_cabinet',  
#     'bench', 'cabinet']
RECON_3D_CLS_OR_dict = {OR: [OR4XCLASSES_dict[OR].index(x) for x in OR4XCLASSES_dict[OR] if x not in OR4XCLASSES_not_detect_dict[OR]] for OR in ['OR42', 'OR45', 'OR46']}

# # -------- OR to NYU40
# OR42_TO_NYU40_CLS_MAPPING = {40:1, 41:2, 24:3, 39:3, 15:4, 18:5, 8:6, 4:7, 38:8, 27:9, 7:10, 4:12, 4:14, 44:2, 28:3, 42:3, 18:4, 21:5, 11:6, 4:7, 41:8, 31:9, 10:10, 7:12, 5:14, 1:16, 16:18, 45:22, 24:30, 32:35, 25:36, 37:37, 8:38, 26:38, \
#     3:39, 6:39, 13:39, 14:39, 40:39, \
#     9:40, 12:40, 15:40, 17:40, 19:40, 20:40, 22:40, 23:40, 27:40, 30:40, 33:40, 34:40, 35:40, 36:40, 38:40, 39:40, 2:0, 29:0}
OR45_TO_NYU40_CLS_MAPPING = {0:0, 43:1, 44:2, 28:3, 42:3, 18:4, 21:5, 11:6, 4:7, 41:8, 31:9, 10:10, 7:12, 5:14, 1:16, 16:18, 45:22, 24:30, 32:35, 25:36, 37:37, 8:38, 26:38, \
    3:39, 6:39, 13:39, 14:39, 40:39, \
    9:40, 12:40, 15:40, 17:40, 19:40, 20:40, 22:40, 23:40, 27:40, 30:40, 33:40, 34:40, 35:40, 36:40, 38:40, 39:40, 2:0, 29:0}



number_pnts_on_template = 2562

# pix3d_n_classes = 9

cls_reg_ratio = 10
obj_cam_ratio = 1

class Dataset_Config(object):
    def __init__(self, dataset, opt=None, OR=None, version='V3', task_name=None, paths={}):
        """
        Configuration of data paths.
        """
        self.dataset = dataset
        if opt is not None and opt.logger is not None:
            self.logger = opt.logger
        else:
            self.logger = basic_logger()


        if self.dataset == 'sunrgbd':
            self.metadata_path = './data/sunrgbd'
            self.train_test_data_path = os.path.join(self.metadata_path, 'sunrgbd_train_test_data')
            self.size_avg_file = os.path.join(self.metadata_path, 'preprocessed/size_avg_category.pkl')
            self.layout_avg_file = os.path.join(self.metadata_path, 'preprocessed/layout_avg_file.pkl')
            self.bins = self.__initiate_bins()
            self.evaluation_path = './evaluation/sunrgbd'
            if not os.path.exists(self.train_test_data_path):
                os.mkdir(self.train_test_data_path)
        if self.dataset == 'OR':
            assert OR is not None
            assert OR in ['OR45']
            assert version in ['V3', 'V4full', 'V4full-detachEmitter'], 'Wrong version: '+version
            self.OR = OR
            # self.data_RAW_path = Path('/newfoundland2/ruizhu/siggraphasia20dataset/layout_labels_%s'%version)
            # self.process_data_root = Path('utils_OR/openrooms') / ('list_%s_%s'%(dataset, version))
            # self.process_data_root = Path(opt.cfg.PATH.total3D_lists_path) / ('list_%s_%s'%(dataset, version))
            self.process_data_root = Path(opt.cfg.PATH.total3D_lists_path) if opt is not None else Path(paths['total3D_lists_path'])
            self.avg_file_path = self.process_data_root / ('preprocessed_%s'%self.OR)
            self.size_avg_file = self.avg_file_path / ('size_avg_category_%s.pkl'%(self.OR))
            self.layout_avg_file = self.avg_file_path / ('layout_avg_file_%s.pkl'%(self.OR))

            self.split_file_path = self.process_data_root / 'list'

            # self.train_test_data_path = '/data/ruizhu/OR-%s-%s_total3D_train_test_data'%(version, self.OR)
            self.train_test_data_path = opt.cfg.DATASET.layout_emitter_path if opt is not None else Path(paths['layout_emitter_path'])

            if os.path.exists(self.layout_avg_file):
                self.bins = self.__initiate_bins()
                # self.evaluation_path = './evaluation/OR_%s_%s'%(version, self.OR)
                # if task_name is not None:
                #     self.evaluation_path += '_%s'%task_name
                # if not os.path.exists(self.train_test_data_path):
                #     os.mkdir(self.train_test_data_path)
            else:
                self.logger.error(red('[!!!] self.layout_avg_file does not exist! %s'%self.layout_avg_file))
                raise ValueError(red('[!!!] self.layout_avg_file does not exist! %s'%self.layout_avg_file))


    def __initiate_bins(self):
        bin = {}

        if self.dataset == 'sunrgbd' or self.dataset == 'OR':
            # there are faithful priors for layout locations, we can use it for regression.
            if os.path.exists(self.layout_avg_file):
                with open(self.layout_avg_file, 'rb') as file:
                    layout_avg_dict = pickle.load(file)
            else:
                raise IOError('No layout average file in %s. Please check.' % (self.layout_avg_file))

            bin['layout_centroid_avg'] = layout_avg_dict['layout_centroid_avg']
            bin['layout_coeffs_avg'] = layout_avg_dict['layout_coeffs_avg']

            '''layout orientation bin'''
            NUM_LAYOUT_ORI_BIN = 2
            ORI_LAYOUT_BIN_WIDTH = np.pi / 4
            bin['layout_ori_bin'] = [[np.pi / 4 + i * ORI_LAYOUT_BIN_WIDTH, np.pi / 4 + (i + 1) * ORI_LAYOUT_BIN_WIDTH] for i in range(NUM_LAYOUT_ORI_BIN)]

            '''camera bin'''
            PITCH_NUMBER_BINS = 2
            PITCH_WIDTH = 40 * np.pi / 180
            ROLL_NUMBER_BINS = 2
            ROLL_WIDTH = 20 * np.pi / 180

            # pitch_bin = [[-60 * np.pi/180, -20 * np.pi/180], [-20 * np.pi/180, 20 * np.pi/180]]
            bin['pitch_bin'] = [[-60.0 * np.pi / 180 + i * PITCH_WIDTH, -60.0 * np.pi / 180 + (i + 1) * PITCH_WIDTH] for
                                i in range(PITCH_NUMBER_BINS)]
            # roll_bin = [[-20 * np.pi/180, 0 * np.pi/180], [0 * np.pi/180, 20 * np.pi/180]]
            bin['roll_bin'] = [[-20.0 * np.pi / 180 + i * ROLL_WIDTH, -20.0 * np.pi / 180 + (i + 1) * ROLL_WIDTH] for i in
                               range(ROLL_NUMBER_BINS)]

            '''bbox orin, size and centroid bin'''
            # orientation bin
            NUM_ORI_BIN = 6
            ORI_BIN_WIDTH = float(2 * np.pi / NUM_ORI_BIN) # 60 degrees width for each bin.
            # orientation bin ranges from -np.pi to np.pi.
            bin['ori_bin'] = [[(i - NUM_ORI_BIN / 2) * ORI_BIN_WIDTH, (i - NUM_ORI_BIN / 2 + 1) * ORI_BIN_WIDTH] for i
                              in range(NUM_ORI_BIN)]

            if os.path.exists(self.size_avg_file):
                with open(self.size_avg_file, 'rb') as file:
                    avg_size = pickle.load(file)
            else:
                raise IOError('No object average size file in %s. Please check.' % (self.size_avg_file))

            bin['avg_size'] = np.vstack([avg_size[key] for key in range(len(avg_size))])

            # for each object bbox, the distance between camera and object centroid will be estimated.

            NUM_DEPTH_BIN = 6
            DEPTH_WIDTH = 1.0
            # centroid_bin = [0, 6]
            bin['centroid_bin'] = [[i * DEPTH_WIDTH, (i + 1) * DEPTH_WIDTH] for i in
                                   range(NUM_DEPTH_BIN)]
        else:
            raise NameError('Please specify a correct dataset name.')

        return bin
