import os.path as osp

import torch
import torchvision

model_url = '/home/hungnd/pytorch-fcn/vgg16_from_caffe.pth'

def VGG16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=True)
    if not pretrained:
        return model
    # model_file = _get_vgg16_pretrained_model()
    state_dict = torch.load(model_url)
    model.load_state_dict(state_dict)
    return model