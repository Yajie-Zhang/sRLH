import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
import numpy as np
import os

pretrain_path='./models/pretrained/resnet18-5c106cde.pth'

class MainNet(nn.Module):
    def __init__(self, num_classes, channels):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.pretrained_model = resnet.resnet18(pretrained=True, pth_path=pretrain_path)
        self.rawcls_net = nn.Linear(channels, num_classes)

    def forward(self, x):
        fm, embeding = self.pretrained_model(x)
        raw_logits = self.rawcls_net(embeding)
        return raw_logits

    def snapshot(self,data,iteration):
        torch.save({
            'iteration':iteration,
            'model_state_dict':self.state_dict(),
        },os.path.join('./checkpoint', '{}_model_ft_{}.t'.format(data,iteration)))

    def load_snapshot(self,root):
        checkpoint=torch.load(root)
        self.load_state_dict(checkpoint['model_state_dict'])
