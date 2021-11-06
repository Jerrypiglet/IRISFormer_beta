import numpy as np
import torch
from utils import transform


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def get_transform_semseg(split, opt, pad_op_override=None):
    assert split in ['train', 'val', 'test']
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    if split == 'train':
        transform_semseg_list_train = [
            # transform.RandScale([opt.semseg_configs.scale_min, opt.semseg_configs.scale_max]),
            # transform.RandRotate([opt.semseg_configs.rotate_min, opt.semseg_configs.rotate_max], padding=mean, ignore_label=opt.semseg_configs.ignore_label),
            # transform.RandomGaussianBlur(),
            # transform.RandomHorizontalFlip(),
            transform.Crop([opt.semseg_configs.train_h, opt.semseg_configs.train_w], crop_type='rand', padding=mean, ignore_label=opt.semseg_configs.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ]
        train_transform = transform.Compose(transform_semseg_list_train)
        return train_transform
    else:
        transform_semseg_list_val = [
            transform.Crop([opt.semseg_configs.train_h, opt.semseg_configs.train_w], crop_type='center', padding=mean, ignore_label=opt.semseg_configs.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ]
        val_transform = transform.Compose(transform_semseg_list_val)
    return val_transform

def get_transform_resize(split, opt, pad_op_override=None, if_gamma_to_hdr=False):
    assert split in ['train', 'val', 'test']
    value_scale = 255
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    mean = [0.5] * 3
    std = [0.5] * 3
    if opt.cfg.MODEL_BRDF.DPT_baseline.enable:
        # following DPT's normalization for depth, at https://github.com/isl-org/DPT/blob/main/run_monodepth.py#L43
        mean = [0.5] * 3
        std = [0.5] * 3
    mean = [item * value_scale for item in mean]
    std = [item * value_scale for item in std]

    # if_pad = False
    # if opt.cfg.DATA.if_pad_to_32x:
    #     im_width_pad_to = int(np.ceil(opt.cfg.DATA.im_width/32.)*32)
    #     im_height_pad_to = int(np.ceil(opt.cfg.DATA.im_height/32.)*32
    #     im_pad_with = 0
    #     if_pad = True

    if split == 'train':
        transform_resize_list_train = []
        if opt.cfg.DATA.if_augment_train:
            transform_resize_list_train = [
                # transform.RandScale([opt.semseg_configs.scale_min, opt.semseg_configs.scale_max]),
                # transform.RandRotate([opt.semseg_configs.rotate_min, opt.semseg_configs.rotate_max], padding=mean, ignore_label=opt.semseg_configs.ignore_label),
                # transform.RandomGaussianBlur(),
                # transform.RandomHorizontalFlip(),
            ]
            assert False, 'Not implemented!'
        transform_resize_list_train = [
            transform.Resize((opt.cfg.DATA.im_width, opt.cfg.DATA.im_height)), 
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ]
        if opt.if_pad:
            transform_resize_list_train.insert(1, opt.pad_op if pad_op_override is None else pad_op_override)
        if opt.if_resize:
            transform_resize_list_train.insert(1, opt.resize_op)

        train_transform = transform.Compose(transform_resize_list_train)
        return train_transform
    else:
        transform_resize_list_val = [
            transform.Resize((opt.cfg.DATA.im_width, opt.cfg.DATA.im_height)), 
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ]
        if opt.if_pad:
            transform_resize_list_val.insert(1, opt.pad_op if pad_op_override is None else pad_op_override)
        if opt.if_resize:
            transform_resize_list_val.insert(1, opt.resize_op)

        val_transform = transform.Compose(transform_resize_list_val)
    return val_transform


def get_transform_matseg(split, opt, pad_op_override=None):
    assert split in ['train', 'val', 'test']
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    ignore_label = 0

    if split == 'train':
        transform_semseg_list_train = [
            # transform.RandScale([opt.semseg_configs, opt.semseg_configs.scale_max]),
            # transform.RandRotate([opt.semseg_configs.rotate_min, opt.semseg_configs.rotate_max], padding=mean, ignore_label=ignore_label),
            # transform.RandomGaussianBlur(),
            # transform.RandomHorizontalFlip(),
            transform.Crop([opt.semseg_configs.train_h, opt.semseg_configs.train_w], crop_type='rand', padding=mean, ignore_label=ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ]
        if opt.if_pad:
            pad_op = opt.pad_op if pad_op_override is None else pad_op_override
        else:
            pad_op = None
            
        if opt.if_pad and pad_op is not None:
            transform_semseg_list_train.insert(1, opt.pad_op if pad_op_override is None else pad_op_override)
        if opt.if_resize and opt.resize_op is not None:
            transform_semseg_list_train.insert(1, opt.resize_op)

        train_transform = transform.Compose(transform_semseg_list_train)
        return train_transform
    else:
        transform_semseg_list_val = [
            transform.Crop([opt.semseg_configs.train_h, opt.semseg_configs.train_w], crop_type='center', padding=mean, ignore_label=ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ]
        if opt.if_pad:
            transform_semseg_list_val.insert(1, opt.pad_op if pad_op_override is None else pad_op_override)
        if opt.if_resize:
            transform_semseg_list_val.insert(1, opt.resize_op)

        val_transform = transform.Compose(transform_semseg_list_val)
    return val_transform

