import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple

# from lib import score_util

from torchfcn.datasets.voc_valid import pascal

PV = pascal('/home/hungnd/Dataset/VOC/VOCdevkit/VOC2011')
valset = PV.get_dset()
plt.ioff()
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['figure.figsize'] = (12, 12)

cnt = 0

for idx in valset:
    sys.stdout.flush()
    im, label = PV.load_image(idx), PV.load_label(idx)
    im_t, label_t = PV.make_translated_frames(im, label, shift=16, num_frames=6)
    for im, label in zip(im_t, label_t):
        plt.imshow(im)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('valid/Shift16/Image/' + str(cnt) + '.png')
        # plt.show()
        
        plt.imshow(PV.palette(label))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('valid/Shift16/Annot/' + str(cnt) + '.png')

        cnt += 1
