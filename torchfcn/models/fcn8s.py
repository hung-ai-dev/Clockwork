import torch
import time
import os.path as osp
import threading
import fcn
import torch.nn as nn
import threading
from .fcn32s import get_upsampling_weight
import torch.multiprocessing as mp

class FCN8s(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
            path=cls.pretrained_model,
            md5='dbd9bbb3829a3184913bccc74373afbb',
        )

    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        self.stage1 = nn.Sequential(
            # conv1
            self.conv1_1, self.relu1_1, self.conv1_2, self.relu1_2, self.pool1,
            # conv2
            self.conv2_1, self.relu2_1, self.conv2_2, self.relu2_2, self.pool2,
            # conv3
            self.conv3_1, self.relu3_1, self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3, self.pool3,
        )

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        self.stage2 = nn.Sequential(
            self.conv4_1, self.relu4_1, self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3, self.pool4  # 1/16
        )

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.stage3 = nn.Sequential(
            # conv5
            self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3, self.pool5,
            # fc6
            self.fc6, self.relu6, self.drop6,
            # fc7
            self.fc7, self.relu7, self.drop7,
        )

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.stage1(x)
        pool3 = h  # 1/8

        h = self.stage2(h)
        pool4 = h  # 1/16

        h = self.stage3(h) # 1/32

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)


class FCN8sAtOnce(FCN8s):
    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s-atonce_from_caffe.pth')

    def __init__(self, n_class = 21):
        super().__init__(n_class = n_class)
        self.save_stage_1 = None
        self.save_stage_2 = None
        self.save_stage_3 = None

        self.clock_1 = 0
        self.clock_2 = 0
        self.clock_3 = 0

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
            path=cls.pretrained_model,
            md5='bfed4437e941fef58932891217fe6464',
        )

    def pipeline_2_stage(self, x, frame0 = None, method = 'Pipe', rate = (1, 1, 1), threshold = 0.0):
        def pipe1(x):
            self.save_stage_1 = self.stage1(x)
            self.save_stage_2 = self.stage2(self.save_stage_1)
            
        def pipe2(x):
            self.save_stage_3 = self.stage3(x)

        if frame0 is not None:
            self.clock_1 = 0
            self.clock_2 = 0

            pipe1(frame0)
            pipe2(self.save_stage_2)
            
            return

        pipe1(x)

        return self.fuse_score(x, self.save_stage_1, self.save_stage_2, self.save_stage_3)

    def pipeline_3_stage(self, x, frame0 = None, frame1 = None, method = 'Pipe', rate = (1, 1, 1), threshold = 0.0):
        def pipe1(x):
            self.save_stage_1 = self.stage1(x)
            
        def pipe2(x):
            self.save_stage_2 = self.stage2(x)

        def pipe3(x):
            self.save_stage_3 = self.stage3(x)

        if frame1 is not None:
            self.clock_1, self.clock_2, self.clock_3 = 0, 0, 0

            pipe1(frame0)
            pipe2(self.save_stage_1)
            pipe3(self.save_stage_2)

            pipe1(frame1)
            pipe2(self.save_stage_1)
            
            return

        self.clock_1 += 1
        self.clock_2 += 1
        self.clock_3 += 1

        pipe1(x)

        return self.fuse_score(x, self.save_stage_1, self.save_stage_2, self.save_stage_3)
        
    
    def fuse_score(self, x, pool3, pool4, fc7):
        h = self.score_fr(fc7)
        h = self.upscore2(h)
        upscore2 = h  # 1/16
    
        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def forward(self, x):
        h = x
        h = self.stage1(x)
        pool3 = h  # 1/8

        h = self.stage2(h)
        pool4 = h  # 1/16

        h = self.stage3(h) # 1/32
        fc7 = h

        return self.fuse_score(x, pool3, pool4, fc7)

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))