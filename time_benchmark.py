import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, CenterCrop, Normalize, Pad
from torchvision.transforms import ToTensor, ToPILImage, Scale
from torch.autograd import Variable

import scipy.misc as m
import cv2
import PIL.Image 
import numpy as np
import os.path as osp
import sys
import torchfcn
from torchfcn.datasets.voc_valid import pascal
import time

model = torchfcn.models.FCN8sAtOnce(n_class=21).cuda()

def timing(model_name):
    total_time = 0

    model.share_memory()
    
    x = torch.FloatTensor(1, 3, 512, 512).cuda()
    x = Variable(x)
    y = torch.FloatTensor(1, 3, 512, 512).cuda()
    y = Variable(y)
    model.pipeline_3_stage(x, x, x)

    for i in range(100):
        x = torch.FloatTensor(1, 3, 512, 512).cuda()
        x = Variable(x)
    
        t0 = time.time()
        if 'Pipe2' in model_name:
            score = model.pipeline_2_stage(x)
        elif 'Pipe3' in model_name:
            score = model.pipeline_3_stage(x)
        elif 'Adaptive' in model_name:
            score = model.adaptive_clockwork(0.25, y)
        else:
            score = model.forward(x)
        pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        t1 = time.time()
        total_time += t1 - t0
    
    print(model_name, total_time / 100)

if __name__ == '__main__':
    timing('Oracle')
    timing('Pipe2')
    timing('Pipe3')
    timing('Adaptive')