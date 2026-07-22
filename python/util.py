"""This module contains simple helper functions """
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch,random
import numpy as np
from collections import OrderedDict

def group_wise_avg(x, p, img_type):
    x1 = x.view(x.shape[0], x.shape[2] // p, p, x.shape[3],x.shape[4])
    if img_type == 'xyt':
        x2 = x1.mean(dim=1, keepdim = True)
    else:
        x2,_ = x1.max(dim=1, keepdim = True)
    return x2

class Train_Logger():
    def __init__(self,save_path,save_name):
        self.log = None
        self.summary = None
        self.save_path = save_path
        self.save_name = save_name

    def update(self,epoch,train_log):
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        #item.update(val_log)
        #print("\033[0;33mTrain:\033[0m",train_log)
        #print("\033[0;33mValid:\033[0m",val_log)
        self.update_csv(item)
        self.update_tensorboard(item)

    def update_csv(self,item):
        print(self.log)
        tmp = pd.DataFrame(item,index=[0])
        if self.log is not None:
            self.log = self.log.append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/%s.csv' %(self.save_path,self.save_name), index=False)

    def update_tensorboard(self,item):
        if self.summary is None:
            self.summary = SummaryWriter('%s/' % self.save_path)
        epoch = item['epoch']
        for key,value in item.items():
            if key != 'epoch': self.summary.add_scalar(key, value, epoch)

class Test_Logger():
    def __init__(self,save_path,save_name):
        self.log = None
        self.summary = None
        self.save_path = save_path
        self.save_name = save_name

    def update(self,name,log):
        item = OrderedDict({'img_name':name})
        item.update(log)
        #print("\033[0;33mTest:\033[0m",log)
        self.update_csv(item)

    def update_csv(self,item):
        tmp = pd.DataFrame(item,index=[0])
        if self.log is not None:
            self.log = self.log.append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/%s.csv' %(self.save_path,self.save_name), index=False)

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

def setpu_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    random.seed(seed)

def dict_round(dic,num):
    for key,value in dic.items():
        dic[key] = round(value,num)
    return dic

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def adjust_learning_rate(optimizer, epoch, args, step):

    lr = args.lr * (0.5 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
