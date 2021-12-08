import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

dropout = nn.Dropout(0.5)

def cal_local_feature(V,step_w=2,step_h=2,device='cpu'):
    #tensor = tensor.to('cpu')
    batch, channel, height, width = V.shape
    A = torch.sum(V, dim=1, keepdim=False)
    V_coor = fm_coordinate(height, width).to(device)
    block_w = int(width / step_w)
    # block_w = int(width / step_w) + 1
    block_h = int(height / step_h)
    # block_h = int(height / step_h) + 1
    v_l = torch.zeros(batch, block_h*block_w, channel).to(device)
    cur_local = 0
    for i in range(block_h):
        for j in range(block_w):
            x1 = i * step_h
            x2 = min((i + 1) * step_h, height)
            y1 = j * step_w
            y2 = min((j + 1) * step_w, width)
            real_width = y2 - y1
            cur_A = A[:, x1:x2, y1:y2]
            cur_A = cur_A.contiguous().view(batch, -1)
            max_index = cur_A.argmax(dim=1).view(batch, -1)
            x_coor = x1 + torch.floor_divide(max_index,real_width)
            #x_coor = x1 + max_index / real_width
            y_coor = y1 + max_index % real_width
            cur_coor = torch.cat((x_coor, y_coor), dim=1).to(device)  # shape=batch,2
            for b in range(batch):
                cur_v_c = V[b, :, x_coor[b], y_coor[b]].view(1,channel)
                repeat_cur_v_c = cur_v_c.repeat(height * width, 1)
                repeat_cur_v_c=F.normalize(repeat_cur_v_c,dim=1)
                cur_V = V[b, :, :, :]
                cur_V = cur_V.permute(1, 2, 0).view(-1, channel)
                cur_V=F.normalize(cur_V,dim=1)
                T = torch.sum(repeat_cur_v_c*cur_V, dim=1).view(1,-1)
                T=torch.max(T,torch.tensor(0.0).to(device))

                cur_b_coor = cur_coor[b, :].view(1, 2)
                repeat_cur_b_coor = cur_b_coor.repeat(height * width, 1)
                E = torch.sqrt(torch.sum((repeat_cur_b_coor.float() - V_coor.float()) ** 2,dim=1)).view(1,-1)+1
                R=T/E
                R=torch.softmax(R,dim=1).view(1,height,width).detach()

                cur_v_c = V[b, :, :, :]
                cur_v_l=cur_v_c*R
                cur_v_l=torch.sum(cur_v_l,dim=(1,2))
                cur_v_l = dropout(cur_v_l)
                v_l[b,cur_local,:]=cur_v_l
            cur_local = cur_local + 1
    return v_l

def fm_coordinate(height,width):
    basic_height_vector=torch.tensor(np.arange(0,height,1))
    basic_height_vector=basic_height_vector.view(basic_height_vector.shape[0],-1)
    basic_width_vector=torch.tensor(np.arange(0,width,1))
    basic_width_vector=basic_width_vector.view(basic_width_vector.shape[0],-1)
    height_vector=basic_height_vector.repeat(1,width).view(-1,1)
    width_vector=basic_width_vector.repeat(height,1)
    coor_vector=torch.cat((height_vector,width_vector),dim=1)
    return coor_vector

def get_index(dis,num_parts):
    sort_dis=torch.sort(dis,dim=1).indices   #from small to big
    return sort_dis[:,0:num_parts]
