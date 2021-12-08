# coding=utf-8
import torch
import random
import torch.nn as nn
from networks.sRLH import MainNet
from utils.hash_loss import Hash_Loss
from utils.cal_map import mean_average_precision
from utils.triplet_loss import TripleLoss,construct_triplets
from utils.tools import *
from utils.dataset import *
import time
import torch.nn.functional as Fc
import itertools
import os

def get_config():
    config = {
        "info": "[sRLH-CUB]",
        "resize_size": 224,
        "batch_size": 64,
        "dataset": "CUB",
        "epoch": 300,
        "test_map": 15,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda"),
        "bit_list": [16,32,48,64],
        "num_workers": 8,
        "channels": 512,
        "init_lr": 0.001,   #todo should be 0.0001
        "weight_decay": 1e-4,
        "num_parts": 3,
        "m": [0.5],
    }
    config = config_dataset(config)
    return config

def gram_schmidt(A):
    """Gram-schmidt正交化"""
    Q = np.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])  # 减去待求向量在以求向量上的投影
        e = u / (np.linalg.norm(u))  # 归一化
        Q[:, cnt] = e
        cnt += 1
    Q = np.sign(Q)
    return Q


def main(config, bit,margin):
    device = config["device"]
    train_data = CUB(config["dataroot"], is_train=True)
    train_img = train_data.train_img
    train_label = train_data.train_label
    num_train = len(train_label)
    n_select_train=3000
    train_dataloader = read_dataset([train_img, train_label], config["resize_size"], config["batch_size"],
                                    config["dataset"], config["num_workers"], is_train=False, is_shuffle=False)

    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed = random.randint(1, 1000)
    seed_torch(seed)
    print(f'seed:{seed}')
    model = MainNet(num_classes=config["n_class"], channels=config["channels"], bit=bit)

    state_dict = torch.load('checkpoint/cub_ft.t', map_location=device)['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    criterion = nn.CrossEntropyLoss()
    criterion2 = Hash_Loss(bit, 0.1)
    criterion3 = TripleLoss(margin)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=config["init_lr"], momentum=0.9, weight_decay=config["weight_decay"])
    model = model.to(device)  ## 部署在GPU  todo

    # 开始训练

    B = torch.zeros(n_select_train, bit).to(device)
    B_label = torch.ones(n_select_train, config["n_class"]).to(device)

    C= torch.sign(torch.rand(config["n_class"], bit) - 0.5).to(device)
    center_label = torch.tensor(np.arange(0, config["n_class"], 1)).float()
    center_label_oh = one_hot_label(center_label, config["n_class"]).to(device)

    for epoch in range(config["epoch"]):
        model.train()
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        cur_train_img = list(np.array(train_img)[select_samples_index])
        cur_train_label = list(np.array(train_label)[select_samples_index])
        cur_dataloader = read_dataset([cur_train_img, cur_train_label], config["resize_size"],
                                      config["batch_size"], config["dataset"], config["num_workers"], is_train=True,
                                      is_shuffle=True)

        cur_train_label_oh = one_hot_label(torch.tensor(np.array(cur_train_label)), config["n_class"]).to(device)
        S = cur_train_label_oh.matmul(center_label_oh.t())
        S[S > 0] = 1
        S[S < 1] = -1
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r
        loss = 0.0

        if epoch < 200:
            lr = config["init_lr"]
        elif epoch < 400:
            lr = config["init_lr"] * 0.1
        else:
            lr = config["init_lr"] * 0.01
        optimizer.param_groups[0]['lr'] = lr

        for i, (index, img,label) in enumerate(cur_dataloader):
            img = img.to(device)
            label_oh = one_hot_label(label, config["n_class"]).to(device)
            label = label.to(device)
            optimizer.zero_grad()
            y_hat, b, v_l = model(img, config["num_parts"], is_train=True, DEVICE=device)

            L_g = criterion(y_hat, label)
            L_h = criterion2(b, C, S[index, :], device)
            anchor, posi, nega = construct_triplets(v_l, label)
            if anchor is None:
                L_l = torch.tensor(0.0, requires_grad=True).to(device)
            else:
                anchor = Fc.normalize(anchor, dim=2)
                posi = Fc.normalize(posi, dim=2)
                nega = Fc.normalize(nega, dim=2)
                po_dis = (anchor * posi).sum(2)
                ne_dis = (anchor * nega).sum(2)
                L_l = criterion3(po_dis, ne_dis, device)

            B[index, :] = b.data
            B_label[index] = label_oh.data

            if epoch < 2:
                total_loss = L_g
            else:
                total_loss = L_g + L_l + L_h
            loss = loss + total_loss
            total_loss.backward()
            optimizer.step()

        print('epoch:{},bit:{},margin:{},loss:{:.4f}'.format(epoch, bit, margin, loss))
        if epoch > 1:
            C = (B_label.t()) @ B
            C = C / torch.sum(B_label.t(), dim=1, keepdim=True)
            C = torch.tensor(gram_schmidt(C.cpu().numpy())).to(device)
        if (epoch + 1) % 50 == 0:
            model.snapshot(config["dataset"],epoch, bit)
            model.eval()
            test_data = CUB(config["dataroot"], is_train=False)
            test_img = test_data.test_img
            test_label = test_data.test_label
            num_test = len(test_label)
            test_dataloader = read_dataset([test_img, test_label], config["resize_size"], config["batch_size"],
                                           config["dataset"], config["num_workers"], is_train=False, is_shuffle=False)

            test_code = torch.zeros(num_test, bit)
            test_label_oh = torch.zeros(num_test, config["n_class"])
            train_code = torch.zeros(num_train, bit)
            train_label_oh = torch.zeros(num_train, config["n_class"])
            with torch.no_grad():
                for i, (index, img, label) in enumerate(test_dataloader):
                    img = img.to(device)
                    label_oh = one_hot_label(label, config["n_class"]).to(device)
                    cur_B = model(img, config["num_parts"], is_train=False, DEVICE=device)
                    test_code[index, :] = torch.sign(cur_B).cpu()
                    test_label_oh[index, :] = label_oh.cpu()
                for i, (index, img, label) in enumerate(train_dataloader):
                    img = img.to(device)
                    label_oh = one_hot_label(label, config["n_class"]).to(device)
                    cur_B = model(img, config["num_parts"], is_train=False, DEVICE=device)
                    train_code[index, :] = torch.sign(cur_B).cpu()
                    train_label_oh[index, :] = label_oh.cpu()
            mAP = mean_average_precision(test_code, test_label_oh, train_code, train_label_oh, num_train)
            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print('epoch:{},time:{},mAP:{:.4f}'.format(epoch, current_time, mAP))

if __name__ == '__main__':
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        for m in config["m"]:
            main(config, bit,m)
