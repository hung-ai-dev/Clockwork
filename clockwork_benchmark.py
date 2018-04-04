import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, CenterCrop, Normalize, Pad
from torchvision.transforms import ToTensor, ToPILImage, Scale
from torch.autograd import Variable

import PIL.Image
import numpy as np
import sys
import time
from torchfcn.models.fcn8s import FCN8sAtOnce
import torch.multiprocessing as mp

# mp = torch.multiprocessing.get_context('spawn')

model = FCN8sAtOnce(n_class=21).cuda()

def fill_in_3_stage(frame0, frame1):
    model.save_stage_1 = model.stage1(frame0)
    model.save_stage_2 = model.stage2(model.save_stage_1)
    model.save_stage_3 = model.stage3(model.save_stage_2)

    model.save_stage_1 = model.stage1(frame1)
    model.save_stage_2 = model.stage2(model.save_stage_1)

def pipe1(x):
    model.save_stage_1 = model.stage1(x)
        
def pipe2(x):
    model.save_stage_2 = model.stage2(x)

def pipe3(x):
    model.save_stage_3 = model.stage3(x)

def pipeline_3_stage(model, x, frame0 = None, frame1 = None,  method = 'Pipe', rate = (1, 1, 1), threshold = 0.0):
    if frame1 is not None:
        fill_in_3_stage(frame0, frame1)
        return

    p1 = mp.Process(target = pipe1, args = (x, ))
    p2 = mp.Process(target = pipe2, args = (model.save_stage_1, ))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    return model.fuse_score(x, model.save_stage_1, model.save_stage_2, model.save_stage_3)

def timing(model_name, model):
    total_time = 0

    x = torch.FloatTensor(1, 3, 256, 256).cuda()
    x = Variable(x)
    pipeline_3_stage(model, x, x, x)

    for i in range(100):
        x = torch.FloatTensor(1, 3, 256, 256).cuda()
        x = Variable(x)

        start = time.time()
        if 'Pipe2' in model_name:
            pipeline_2_stage(model, x)
        elif 'Pipe3' in model_name:
            pipeline_3_stage(model, x)
        else:
            model.forward(x)
        torch.cuda.synchronize()
        end = time.time()

        total_time += end- start
        x = None

    print(model_name, total_time / 100)

if __name__ == '__main__':
    mp.set_start_method("spawn")

    # model.share_memory()
 
    timing('Oracle', model)
    timing('Pipe3', model)