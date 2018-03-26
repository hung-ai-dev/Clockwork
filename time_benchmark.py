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
from concurrent.futures import ThreadPoolExecutor

import torch
torch.multiprocessing.set_start_method("spawn")

# mp = _mp.get_context('spawn')
model = torchfcn.models.FCN8sAtOnce(n_class=21).cuda()

def timing(model_name):
    total_time = 0

    model.share_memory()

    x = torch.FloatTensor(1, 3, 512, 512).cuda()
    x = Variable(x)
    model.pipeline_2_stage(x, x, x)

    for i in range(100):
        x = torch.FloatTensor(1, 3, 512, 512).cuda()
        x = Variable(x)

        start = time.time()
        if 'Pipe2' in model_name:
            model.pipeline_2_stage(x)
        elif 'Pipe3' in model_name:
            model.pipeline_3_stage(x)
        else:
            model.forward(x)
        torch.cuda.synchronize()
        end = time.time()

        total_time += end- start
    print(model_name, total_time / 100)

timing('Oracle')
timing('Pipe3')
timing('Pipe2')
