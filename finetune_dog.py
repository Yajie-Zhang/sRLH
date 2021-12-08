#coding=utf-8
import torch
import os
import random
import torch.nn as nn
from networks.fine_tune import MainNet
from utils.tools import *
from utils.dataset import *
from utils.read_data import Read_Dataset
import time

def get_config():
    config={
        "alpha":0.1,
        "info":"[sRLH-FT-DOG]",
        "resize_size":224,
        "batch_size":32,
        "dataset":"DOG",   #"AIR", "CAR"
        "epoch":500,
        #"device":torch.device("cpu"),
        "device":torch.device("cuda"),
        "num_workers":8,
        "channels":512,
        "init_lr":0.001,
        "weight_decay":1e-4,
    }
    config=config_dataset(config)
    return config

def main(config):
    device = config["device"]

    train_data = Stanford_Dogs(config["dataroot"], is_train=True)
    train_img_label = train_data.train_img_label
    num_train = len(train_img_label)
    train_label = torch.zeros(num_train)
    for i in range(num_train):
        cur_label = train_img_label[i][1]
        train_label[i] =torch.tensor (cur_label).float()
    train_dataloader, num_train, test_dataloader, num_test = Read_Dataset(config, num_train)

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
    model = MainNet(num_classes=config["n_class"], channels=config["channels"])

    criterion = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=config["init_lr"], momentum=0.9, weight_decay=config["weight_decay"])
    model = model.to(device)  ## 部署在GPU  todo

    for epoch in range(config["epoch"]):
        model.train()
        loss=0.0
        prec=0.0
        for i, (index,img,label) in enumerate(train_dataloader):
            img=img.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            raw_logits= model(img)
            raw_loss = criterion(raw_logits, label)
            total_loss = raw_loss
            loss=loss+total_loss
            total_loss.backward()

            optimizer.step()
            pred = raw_logits.max(1, keepdim=True)[1]
            prec += pred.eq(label.view_as(pred)).sum().item()

        print('epoch:{},loss:{:.4f},prec:{:.4f}'.format(epoch, loss, prec / num_train))
        if epoch%10==0:
            model.snapshot(config["dataset"],epoch)

if __name__ == '__main__':
    config = get_config()
    print(config)
    main(config)
