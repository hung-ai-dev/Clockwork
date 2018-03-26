import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, CenterCrop, Normalize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from torch.autograd import Variable

import scipy.misc as m
import cv2
import PIL.Image
import numpy as np
import os.path as osp
import sys
import torchfcn
from torchfcn.datasets.voc_valid import pascal

cuda = False

root = 'valid'
shift_type = 'Shift16'
data_folder = osp.join(root, shift_type)
resume = 'model_best.pth.tar'

model = torchfcn.models.FCN8sAtOnce(n_class=21).cuda()
checkpoint = torch.load(resume)
# model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
start_iteration = checkpoint['iteration']

def transform(img, lbl, img_size = (512, 512)):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= mean_bgr
    img = m.imresize(img, (512, 512))
    img = img.transpose(2, 0, 1)

    print(lbl.shape)
    lbl = lbl.astype(float)
    lbl[lbl==255] = -1
    lbl = m.imresize(lbl, (512, 512))
    lbl = lbl.astype(int)
    img = torch.from_numpy(img).float()
    lbl = torch.from_numpy(lbl).long()

    return img, lbl

img_file = '/home/hungnd/Dataset/VOC/VOCdevkit/VOC2011/JPEGImages/2011_003271.jpg'
lbl_file = '/home/hungnd/Dataset/VOC/VOCdevkit/VOC2011/SegmentationClass/2011_003271.png'

im = m.imread(img_file)
im = np.array(im, dtype=np.uint8)
lbl = m.imread(lbl_file)
lbl = np.array(lbl, dtype=np.uint8)

img, lbl = transform(im, lbl)
img.unsqueeze_(0)
lbl.unsqueeze_(0)
img, lbl = Variable(img.cuda()), Variable(lbl.cuda())

print(img)

out = model.forward(img)