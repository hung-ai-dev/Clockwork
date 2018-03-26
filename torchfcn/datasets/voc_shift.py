#!/usr/bin/env python3
import collections
import os
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data


class VOCShift(data.Dataset):

    class_names = np.array([
        'background',        'aeroplane',        'bicycle',        'bird',        'boat',
        'bottle',        'bus',        'car',        'cat',        'chair',        'cow',
        'diningtable',        'dog',        'horse',        'motorbike',        'person',
        'potted plant',        'sheep',        'sofa',        'train',        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, shift_type = 'Shift16', transform=False):
        self.root = root
        # self.split = split
        self._transform = transform

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, shift_type)
        self.files = []

        for split in ['train', 'val']:
            imgsets_file = os.listdir(osp.join(dataset_dir, 'Image'))

            for did in imgsets_file:
                did = did.strip()
                img_file = osp.join(dataset_dir, 'Image/%s' % did)
                lbl_file = osp.join(dataset_dir, 'Annot/%s' % did)

                self.files.append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl
