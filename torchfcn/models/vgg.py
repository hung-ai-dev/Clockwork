import os.path as osp

import fcn

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


def _get_vgg16_pretrained_model():
    return fcn.data.cached_download(
        url='http://drive.google.com/uc?id=0B9P1L--7Wd2vLTJZMXpIRkVVRFk',
        path=osp.expanduser('~/data/models/pytorch/vgg16_from_caffe.pth'),
        md5='aa75b158f4181e7f6230029eb96c1b13',
    )
