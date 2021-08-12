import torch
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import logging
from utils.maskrcnn_rui.utils.comm import get_world_size
from utils.maskrcnn_rui.data import samplers
from utils.utils_training import cycle
from utils.utils_misc import white_blue, basic_logger
import math
import numpy as np
import random
from torch.utils.data import BufferedShuffleDataset 
import multiprocessing as mp


def make_data_sampler_binary(dataset, shuffle, opt, distributed, if_distributed_override):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=opt.num_gpus,
            rank=opt.rank,
            shuffle=shuffle
        )

    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(
    sampler, images_per_batch, num_iters=None, start_iter=0, drop_last=True
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=drop_last
    )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def make_data_loader_binary(opt, dataset, is_train=True, start_iter=0, logger=None, override_shuffle=None, collate_fn=None, batch_size_override=-1, workers=-1, pin_memory = True, if_distributed_override=True):
    cfg = opt.cfg
    num_gpus = opt.num_gpus
    if logger is None:
        logger = basic_logger()

    is_distributed=opt.distributed and if_distributed_override
    num_workers = cfg.DATASET.num_workers if workers==-1 else workers
    if is_train:
        images_per_gpu = cfg.SOLVER.ims_per_batch if batch_size_override==-1 else batch_size_override
        shuffle = True
        num_iters = cfg.SOLVER.max_iter
        drop_last = False
        persistent_workers = True
    else:
        images_per_gpu = cfg.TEST.ims_per_batch if batch_size_override==-1 else batch_size_override
        shuffle = False
        num_iters = None
        start_iter = 0
        drop_last = False
        persistent_workers = False

    if override_shuffle is not None:
        shuffle = override_shuffle
        
    sampler = make_data_sampler_binary(dataset, shuffle, opt, distributed=is_distributed, if_distributed_override=if_distributed_override)
    
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, None, start_iter, drop_last=drop_last
    )

    if num_workers <= 0:
        random.seed(134714)
    

    # bf_dataset = BufferedShuffleDataset(dataset, buffer_size=500) # https://pytorch.org/docs/1.8.0/data.html?highlight=bufferedshuffledataset#torch.utils.data.BufferedShuffleDataset
    bf_dataset = dataset

    data_loader = torch.utils.data.DataLoader( # https://pytorch.org/docs/stable/data.html
        bf_dataset,
        num_workers=num_workers,
        batch_size=images_per_gpu, 
        # batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        # shuffle=shuffle,
        persistent_workers=persistent_workers, 
        worker_init_fn=worker_init_fn, 
        # drop_last=is_train, 
        # multiprocessing_context=mp.get_context('fork')
        # multiprocessing_context='forkserver'
    )

    # print('<-----', mp.get_context('fork'))
    logger.info(white_blue('[utils_dataloader] %s-%s with bs %d*%d: len(dataset) %d, len(sampler) %d, len(batch_sampler) %d, is_train %s, is_distributed %s:' % \
                (dataset.dataset_name, dataset.split, images_per_gpu, num_gpus, len(dataset), len(sampler), len(batch_sampler), is_train, is_distributed)))
    return data_loader, images_per_gpu

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    if worker_info.num_workers >= 1:
        random.seed(1143234+worker_id*13)
    # overall_start = dataset.start
    # overall_end = dataset.end
    # # configure the dataset to only process the split workload
    # per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    # worker_id = worker_info.id
    # dataset.start = overall_start + worker_id * per_worker
    # dataset.end = min(dataset.start + per_worker, overall_end)
    # dataset.meta_split_scene_name_list_workers = [list(_) for _ in np.array_split(dataset.meta_split_scene_name_list, worker_info.num_workers)][worker_id]
    # print('>>>>>', worker_id, len(dataset.meta_split_scene_name_list_workers))
