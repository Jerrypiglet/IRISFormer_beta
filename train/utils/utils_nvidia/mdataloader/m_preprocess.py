'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''

import torchvision.transforms as transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],\
        'std': [0.229, 0.224, 0.225]}

def normalize_intensity( normalize_paras_):
    '''
    ToTensor(), Normalize()
    '''
    transform_list = [ transforms.ToTensor(),
           transforms.Normalize(**__imagenet_stats)]
    return transforms.Compose(transform_list)

def to_tensor():
    return transforms.ToTensor()


def get_transform():
    '''
    API to get the transformation. 
    return a list of transformations
    '''
    transform_list = normalize_intensity(__imagenet_stats)
    return transform_list

def get_transform_th():
    transform_list = transforms.Compose([ transforms.Normalize(**__imagenet_stats)])

    return transform_list
