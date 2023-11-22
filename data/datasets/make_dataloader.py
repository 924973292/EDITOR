import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from .ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
from .bases import ImageDataset
from .sampler import RandomIdentitySampler, IdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .msvr310 import MSVR310
from .RGBNT201 import RGBNT201
from .RGBNT100 import RGBNT100
from .RGBNT300 import RGBNT300
from .RegDB import RegDB
from .SYSU import SYSU
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
import torch.utils.data as data
from data.cross.data_loader import SYSUData, RegDBData, TestData
from data.cross.data_manager import process_query_sysu, process_gallery_sysu, process_test_regdb
import time
import torchvision.transforms as transforms
from itertools import product

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'RGBNT201': RGBNT201,
    'RGBNT100': RGBNT100,
    'MSVR310': MSVR310,
    'RegDB': RegDB,
    'SYSU': SYSU,
    'RGBNT300':RGBNT300,
}
""" Random Erasing (Cutout)

Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2019, Ross Wightman
"""
import random
import math

import torch


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5,
            min_area=0.02,
            max_area=1 / 3,
            min_aspect=0.3,
            max_aspect=None,
            mode='const',
            min_count=1,
            max_count=None,
            num_splits=0,
            device='cuda',
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input

    def __repr__(self):
        # NOTE simplified state for repr
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs


def train_collate_fn_Cross(batch):
    # Group images by pids and camid
    imgs, pids, camids, viewids, img_path = zip(*batch)
    imgs = list(imgs)
    pids = list(pids)
    camids = list(camids)
    viewids = list(viewids)
    grouped_data = {}
    for img, pid, camid, viewid, _ in batch:
        key = pid
        if key not in grouped_data:
            grouped_data[key] = {'RGB': [], 'NI': [], 'TI': []}
        grouped_data[key]['RGB'].append(img[0])
        grouped_data[key]['NI'].append(img[1])
        grouped_data[key]['TI'].append(img[2])

    # Randomly shuffle the modality sequences within each pid
    for key, modality_data in grouped_data.items():
        rgb_data = modality_data['RGB']
        ni_data = modality_data['NI']
        ti_data = modality_data['TI']
        combinations = product(rgb_data, ni_data, ti_data)
        imgs.extend(combinations)
        pids.extend([key] * len(rgb_data) * len(ni_data) * len(ti_data))

    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, torch.tensor(pids), torch.tensor(camids), torch.tensor(viewids), img_path


def train_collate_fn_RegDB(batch):
    # Group images by pids and camid
    imgs, pids, camids, viewids, imgpath = zip(*batch)
    imgs = list(imgs)
    pids = list(pids)
    camids = list(camids)
    viewids = list(viewids)
    grouped_data = {}
    for img, pid, camid, viewid, _ in batch:
        key = (pid, camid)
        if key not in grouped_data:
            grouped_data[key] = {'RGB': [], 'NI': []}
        grouped_data[key]['RGB'].append(img[0])
        grouped_data[key]['NI'].append(img[1])

    # Randomly shuffle the modality sequences within each pid
    for key, modality_data in grouped_data.items():
        length = len(modality_data['RGB'])
        for i in range(length):
            for j in range(length):
                if j != i:
                    imgs.append(
                        [modality_data['RGB'][i], modality_data['NI'][j]])
                    pids.append(key[0])
                    camids.append(key[1])

    RGB_list = []
    NI_list = []
    TI_list = []
    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[1])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)

    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, torch.tensor(pids), torch.tensor(camids), torch.tensor(viewids), imgpath


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, imgpath = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        if len(img) == 2:
            RGB_list.append(img[0])
            NI_list.append(img[1])
            TI_list.append(img[1])
        else:
            RGB_list.append(img[0])
            NI_list.append(img[1])
            TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, pids, camids, viewids, imgpath


def val_collate_fn_RegDB(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)

    imgs = {'RGB': RGB, "NI": NI}
    return imgs, pids, camids, camids_batch, viewids, img_paths


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        if len(img) == 2:
            RGB_list.append(img[0])
            NI_list.append(img[1])
            TI_list.append(img[1])
        else:
            RGB_list.append(img[0])
            NI_list.append(img[1])
            TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # T.ColorJitter(brightness=[0.8, 1.2],contrast=[0.85, 1.15])
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH,
                                                     cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers,
                collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    if cfg.DATASETS.NAMES == 'RegDB' or cfg.DATASETS.NAMES == 'SYSU':
        val_set = ImageDataset(dataset.query, val_transforms)
    else:
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
