import torch
from torch.autograd import Variable
import numpy as np
import time
import cv2
import PIL.Image 
import numpy as np
import os.path as osp
import sys
import torchfcn
from torchfcn.datasets.voc_valid import pascal
from torchfcn.datasets.youtube import youtube

resume = 'model_best.pth.tar'

model = torchfcn.models.FCN8sAtOnce(n_class=21).cuda()
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['model_state_dict'])

YT = youtube('/home/hungnd/Dataset/youtube_masks')
PV = pascal('/home/hungnd/Dataset/VOC/VOCdevkit/VOC2011')
n_cl = len(YT.classes)
inputs = YT.load_dataset()

def transform(img, lbl, img_size = (512, 512)):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    img = np.array(img)
    lbl = np.array(lbl)

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

def benchmark(model_name, thresh):
    offset = 10
    skip = 2
    if 'Pipe3' in model_name:
        skip = 3

    label_trues, label_preds = [], []
    total_time = 0
    cnt = 0

    f_2 = 0
    f_1 = 0
    f_0 = 0

    for (class_, vid, shot) in inputs:
        model.prev_scores = None

        for f in YT.list_label_frames(class_, vid, shot):
            print(class_, vid, shot, f)

            f_2 = f_1
            f_1 = f_0
            f_0 = f

            if f < skip*offset + 1:
                continue

            if 'Pipe2' in model_name:
                im = YT.load_frame(class_, vid, shot, f_1)
                lbl = YT.make_label(YT.load_label(class_, vid, shot, f_1), class_)
                
                img, label = transform(im, lbl)
                img = Variable(img.cuda())

                model.pipeline_2_stage(None, img)
            elif 'Pipe3' in model_name:
                im0 = YT.load_frame(class_, vid, shot, f_2)
                lbl0 = YT.make_label(YT.load_label(class_, vid, shot, f_2), class_)
                
                img0, label0 = transform(im0, lbl0)
                img0 = Variable(img0.cuda())

                im1 = YT.load_frame(class_, vid, shot, f_1)
                lbl1 = YT.make_label(YT.load_label(class_, vid, shot, f_1), class_)
                
                img1, label1 = transform(im1, lbl1)
                img1 = Variable(img1.cuda())

                model.pipeline_3_stage(None, img0, img1)
            
            im = YT.load_frame(class_, vid, shot, f)
            lbl = YT.make_label(YT.load_label(class_, vid, shot, f), class_)
            
            img, label = transform(im, lbl)
            img = Variable(img.cuda())
            
            t0 = time.time()
            cnt += 1
            if model_name == 'Oracle':
                score = model(img)
            elif 'Pipe2' in model_name:
                score = model.pipeline_2_stage(img)
            elif 'Pipe3' in model_name:
                score = model.pipeline_3_stage(img)
            elif 'Adaptive' in model_name:
                score = model.adaptive_clockwork(thresh, img)
            out = score.data.max(1)[1].cpu().numpy()[:, :, :]

            t1 = time.time()
            total_time += t1 - t0

            lbl_pred = np.zeros(out.shape, dtype=np.uint8)
            for c in YT.classes:
                lbl_pred[out == PV.classes.index(c)] = YT.classes.index(c)
          
            label_preds.append(lbl_pred)
            label_trues.append(np.array(lbl))
        
    acc, acc_cls, mean_iu, fwavacc = torchfcn.utils.label_accuracy_score(label_trues, label_preds, n_class=n_cl)
    
    file_name = model_name + '.txt'
    f = open(file_name, 'w')
    f.write('\nTime ' + str(total_time / cnt))
    f.write('\nAcc ' + str(acc))
    f.write('\nAcc_cls ' + str(acc_cls))
    f.write('\nMeanIoU ' + str(mean_iu))
    f.write('\nFwavacc ' + str(fwavacc) + '\n')

if __name__ == '__main__':
    benchmark('Pipe3', 0)

    thresh = [0.15, 0.25, 0.35, 0.45]
    for th in thresh:
        benchmark('Adaptive_' + str(th), th)

    benchmark('Oracle', 0)    