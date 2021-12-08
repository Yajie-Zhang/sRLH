import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms
import torch
from scipy.io import loadmat

class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

class STANFORD_CAR():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'cars_train')
        test_img_path = os.path.join(self.root, 'cars_test')
        train_label_file = open(os.path.join(self.root, 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]

class FGVC_aircraft():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'data', 'images')
        test_img_path = os.path.join(self.root, 'data', 'images')
        train_label_file = open(os.path.join(self.root, 'data', 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'data', 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]

class Stanford_Dogs():
    def __init__(self,root, is_train=True):
        self.root=root
        self.is_train=is_train
        train_path=loadmat(self.root+'/train_data.mat')['train_info'][0][0]
        train_img_path=train_path[0]
        num_train=train_img_path.shape[0]
        train_label_path=train_path[2]
        test_path=loadmat(self.root+'/test_data.mat')['test_info'][0][0]
        test_img_path=test_path[0]
        num_test=test_img_path.shape[0]
        test_label_path=test_path[2]
        train_img_label=[]
        test_img_label=[]
        for i in range(num_train):
            train_img_label.append([self.root+'/images/Images/'+train_img_path[i,0][0], train_label_path[i,0]-1])
        for i in range(num_test):
            test_img_label.append([self.root+'/images/Images/'+test_img_path[i,0][0], test_label_path[i,0]-1])
        self.train_img_label=train_img_label
        self.test_img_label=test_img_label


def train_transforms(img,input_size):
    img = transforms.Resize((input_size,input_size), Image.BILINEAR)(img)
    img = transforms.RandomHorizontalFlip()(img)
    img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    return img

class CUB_DATASET():
    def __init__(self, imgs, labels, input_size,is_train=True):
        self.imgs = imgs
        self.labels = labels
        self.input_size = input_size
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.imgs[index]), self.labels[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            original_img=train_transforms(img,self.input_size)
            return index,original_img,target


        else:
            img, target = imageio.imread(self.imgs[index]), self.labels[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((self.input_size,self.input_size), Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            return index, img, target

    def __len__(self):
        return len(self.labels)

class DATASET():
    #CAR and AIR dataset
    def __init__(self,data, input_size,is_train=True):
        self.data=data
        self.input_size = input_size
        self.is_train = is_train


    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.data[index][0]), int(self.data[index][1])
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            original_img = train_transforms(img, self.input_size)
            return index, original_img, target

        else:
            img, target = imageio.imread(self.data[index][0]), self.data[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size,self.input_size), Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            return index,img, target

    def __len__(self):
        return len(self.data)

def read_dataset(data,input_size, batch_size,set,num_workers=8, is_train=True,is_shuffle=True):
    if set=='CUB':
        dataset = CUB_DATASET(data[0], data[1], input_size, is_train)
    else:
        dataset=DATASET(data,input_size,is_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle,num_workers=num_workers,drop_last=False)
    return dataloader

