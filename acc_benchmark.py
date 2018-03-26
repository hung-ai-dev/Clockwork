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

cuda = False

root = 'valid'
shift_type = 'Shift16'
data_folder = osp.join(root, shift_type)
resume = 'model_best.pth.tar'

model = torchfcn.models.FCN8sAtOnce(n_class=21).cuda()
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
start_iteration = checkpoint['iteration']

def transform(img, lbl, img_size = (512, 512)):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= mean_bgr
    # img = Scale(512, )
    img = img.transpose(2, 0, 1)
    
    lbl = lbl.astype(float)
    lbl[lbl==255] = -1
    # lbl = m.imresize(lbl, (512, 512))
    lbl = lbl.astype(int)
    img = torch.from_numpy(img).float()
    lbl = torch.from_numpy(lbl).long()
    
    return img.unsqueeze_(0), lbl.unsqueeze_(0)

PV = pascal('/home/hungnd/Dataset/VOC/VOCdevkit/VOC2011')
valset = PV.get_dset()


def score_translations(model_name, model, shift, num_frames):
    label_trues, label_preds = [], []

    offset = 0

    if 'Pipe2' in model_name:
        offset = 2
    elif 'Pipe3' in model_name:
        offset = 3

    cnt = -1
    for idx in valset:
#        cnt += 1
#        if cnt == 10:
#           break
        sys.stdout.flush()
        im, label = PV.load_image(idx), PV.load_label(idx)
        im_t, label_t = PV.make_translated_frames(im, label, shift=16, num_frames=6)
        sys.stdout.flush()
        im, label = PV.load_image(idx), PV.load_label(idx)
        im_frames, label_frames = PV.make_translated_frames(im, label, shift=shift, num_frames=num_frames)

        frame0, label0 = transform(im_frames[0], label_frames[0])
        frame1, label1 = transform(im_frames[1], label_frames[1])
        frame2, label2 = transform(im_frames[2], label_frames[2])
        
        frame0 = Variable(frame0.cuda())
        frame1 = Variable(frame1.cuda())
        frame2 = Variable(frame2.cuda())

        if 'Pipe2' in model_name:
            model.pipeline_2_stage(x = None, frame0 = frame0, frame1 = frame1)
        if 'Pipe3' in model_name:
            model.pipeline_3_stage(x = None, frame0 = frame0, frame1 = frame1, frame2 = frame2)

        im_frames, label_frames = im_frames[offset:], label_frames[offset:]

        for data, annot in zip(im_frames, label_frames):
            data, target = transform(data, annot)
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            if model_name == 'Oracle':
                score = model(data)
            elif 'Pipe2' in model_name:
                score = model.pipeline_2_stage(data)
            elif 'Pipe3' in model_name:
                score = model.pipeline_3_stage(data)

            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()

            label_preds.append(lbl_pred)
            label_trues.append(annot)

    acc, acc_cls, mean_iu, fwavacc = torchfcn.utils.label_accuracy_score(label_trues, label_preds, n_class=21)
    
    file_name = model_name + '_' + str(shift) + '.txt'
    f = open(file_name, 'w')
    f.write('\nAcc' + str(acc))
    f.write('\nAcc_cls' + str(acc_cls))
    f.write('\nMeanIoU' + str(mean_iu))
    f.write('\nFwavacc' + str(fwavacc) + '\n')

score_translations('Oracle', model, 16, 6)
score_translations('Oracle', model, 32, 6)
