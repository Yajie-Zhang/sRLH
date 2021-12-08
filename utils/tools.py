import torch
import numpy as np
import os

user=os.getenv('SLURM_JOB_USER')
job=os.getenv('SLURM_JOB_ID')
#tmp='/ssd/'+user+'/'+job+'/fine_grained_dataset'
#tmp='/home/yajie/文档/fine_grained_dataset'
#tmp='/mnt/10501001/fine_grained_dataset'
tmp='/home/data_10501005'

def one_hot_label(label,num_class):
    label=label.view(-1,1).long().cpu()
    onehot=torch.zeros(label.shape[0],num_class)
    onehot=onehot.scatter(1,label,1)
    return onehot

def config_dataset(config):
    if config["dataset"]=="CUB":
        config["dataroot"] = tmp+'/CUB_200_2011'
        config["n_class"]=200
    elif config["dataset"]=="AIR":
        config["dataroot"]=tmp+'/FGVC-aircraft'
        config["n_class"]=100
    elif config["dataset"]=="CAR":
        config["dataroot"]=tmp+'/Stanford_Cars'
        config["n_class"]=196
    else:
        config["dataroot"]=tmp+'/dogs'
        config["n_class"]=120
    return config

