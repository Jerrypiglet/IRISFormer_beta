class Relation_Config(object):
    def __init__(self):
        self.d_g = 64
        self.d_k = 64
        self.Nr = 16

import os
import numpy as np
import pickle
from pathlib import Path

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

RECON_3D_CLS_OR_dict = {OR: [OR4XCLASSES_dict[OR].index(x) for x in OR4XCLASSES_dict[OR] if x not in OR4XCLASSES_not_detect_dict[OR]] for OR in ['OR42', 'OR45', 'OR46']}

OR4X_mapping_catInt_to_RGB = {'light': 'OR4X_mapping_catInt_to_RGB_light.pkl', 'dark': 'OR4X_mapping_catInt_to_RGB_dark.pkl'}
OR4X_mapping_catStr_to_RGB = {'light': 'OR4X_mapping_catStr_to_RGB_light.pkl', 'dark': 'OR4X_mapping_catStr_to_RGB_dark.pkl'}
