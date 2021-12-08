import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
import numpy as np
import os

from utils.sRLM import cal_local_feature,get_index

pretrain_path='./models/pretrained/resnet18-5c106cde.pth'

class MainNet(nn.Module):
    def __init__(self, num_classes, channels,bit):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.pretrained_model = resnet.resnet18(pretrained=True, pth_path=pretrain_path)
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.hash_layer=nn.Sequential(
            nn.Linear(channels,bit),
            nn.Tanh()
        )

    def forward(self, x,num_parts,is_train=True,DEVICE='cuda'):
        if is_train==True:
            V, g = self.pretrained_model(x)  #fm_shape=batch,512,7,7  g.shape=batch,512
            batch=V.shape[0]
            y_hat = self.rawcls_net(g)
            hash_code=self.hash_layer(g)

            v_l = cal_local_feature(V,device=DEVICE)  #shape=batch,num_parts*num_parts,512
            repeat_g = g.repeat(1, num_parts*num_parts)
            repeat_g = repeat_g.view(v_l.shape)
            dis = torch.sum(torch.sqrt((repeat_g - v_l) ** 2), dim=2)
            select_index = get_index(dis, num_parts)

            select_v_l = torch.zeros(batch, num_parts, g.shape[1]).to(DEVICE)
            for i in range(batch):
                select_v_l[i, :, :] = v_l[i, select_index[i, :], :]
            select_v_l=select_v_l.view(batch,num_parts,-1)
            return y_hat, hash_code,select_v_l
        else:
            V, g = self.pretrained_model(x)
            hash_code = self.hash_layer(g)
            return hash_code

    def snapshot(self,data,iteration,bit):
        torch.save({
            'iteration':iteration,
            'model_state_dict':self.state_dict(),
        },os.path.join('./checkpoint', '{}_model_{}_{}.t'.format(data,iteration,bit)))

    def load_snapshot(self,root):
        checkpoint=torch.load(root)
        self.load_state_dict(checkpoint['model_state_dict'])
