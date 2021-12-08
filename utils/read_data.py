from utils.dataset import *

def Read_Dataset(config,n_select_train):
    if config["dataset"]=="CUB":
        train_data=CUB(config["dataroot"],is_train=True)
        train_img=train_data.train_img
        train_label=train_data.train_label
        num_train=len(train_label)
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        select_train_img=list(np.array(train_img)[select_samples_index])
        select_train_label=list(np.array(train_label)[select_samples_index])
        train_dataloader=read_dataset([select_train_img,select_train_label],config["resize_size"],config["batch_size"],config["dataset"],config["num_workers"],is_train=True,is_shuffle=True)
        test_data=CUB(config["dataroot"],is_train=False)
        test_img=test_data.test_img
        test_label=test_data.test_label
        num_test=len(test_label)
        test_dataloader=read_dataset([test_img,test_label],config["resize_size"],config["batch_size"],config["dataset"],config["num_workers"],is_train=False,is_shuffle=False)
        #return train_dataloader,num_train,test_dataloader,num_test
    elif config["dataset"]=="AIR":
        train_data=FGVC_aircraft(config["dataroot"],is_train=True)
        train_img_label=train_data.train_img_label
        num_train=len(train_img_label)
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        select_img_label=list(np.array(train_img_label)[select_samples_index])
        train_dataloader=read_dataset(select_img_label,config["resize_size"],config["batch_size"],config["dataset"],config["num_workers"],is_train=True,is_shuffle=True)
        test_data=FGVC_aircraft(config["dataroot"],is_train=False)
        test_img_label=test_data.test_img_label
        num_test=len(test_img_label)
        test_dataloader=read_dataset(test_img_label,config["resize_size"],config["batch_size"],config["dataset"],config["num_workers"],is_train=False,is_shuffle=False)
        #return train_dataloader,num_train,test_dataloader,num_test
    elif config["dataset"]=="CAR":
        train_data=STANFORD_CAR(config["dataroot"],is_train=True)
        train_img_label=train_data.train_img_label
        num_train=len(train_img_label)
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        select_img_label = list(np.array(train_img_label)[select_samples_index])
        train_dataloader=read_dataset(select_img_label,config["resize_size"],config["batch_size"],config["dataset"],config["num_workers"],is_train=True,is_shuffle=True)
        test_data=STANFORD_CAR(config["dataroot"],is_train=False)
        test_img_label=test_data.test_img_label
        num_test=len(test_img_label)
        test_dataloader=read_dataset(test_img_label,config["resize_size"],config["batch_size"],config["dataset"],config["num_workers"],is_train=False,is_shuffle=False)
        #return train_dataloader,num_train, test_dataloader,num_test
    else:
        train_data = Stanford_Dogs(config["dataroot"], is_train=True)
        train_img_label = train_data.train_img_label
        num_train = len(train_img_label)
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        select_img_label = list(np.array(train_img_label)[select_samples_index])
        train_dataloader = read_dataset(select_img_label, config["resize_size"], config["batch_size"],
                                        config["dataset"], config["num_workers"], is_train=True, is_shuffle=True)
        test_data = Stanford_Dogs(config["dataroot"], is_train=False)
        test_img_label = test_data.test_img_label
        num_test = len(test_img_label)
        test_dataloader = read_dataset(test_img_label, config["resize_size"], config["batch_size"], config["dataset"],
                                       config["num_workers"], is_train=False, is_shuffle=False)
        # return train_dataloader,num_train, test_dataloader,num_test
    return train_dataloader, num_train, test_dataloader, num_test
