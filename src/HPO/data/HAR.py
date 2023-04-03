import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class HAR(Dataset):

  def __init__(self, train: bool, augmentation = None, cuda_device = 0,**kwargs):
    PATH = "/home/cmackinnon/scripts/datasets/HAR/UCI HAR Dataset/"
    self.cuda_device = cuda_device
    if train:
      PATH_x = "/home/cmackinnon/scripts/datasets/HAR/train_x.npy"#UCI HAR Dataset/train/Inertial Signals"
      PATH_y = "/home/cmackinnon/scripts/datasets/HAR/UCI HAR Dataset/train/y_train.txt"
    else:
      PATH_x = "/home/cmackinnon/scripts/datasets/HAR/test_x.npy"#UCI HAR Dataset/test/Inertial Signals"
      PATH_y = "/home/cmackinnon/scripts/datasets/HAR/UCI HAR Dataset/test/y_test.txt"
    #data = np.swapaxes(data,1,2)
    
    self.x = torch.from_numpy(np.load(PATH_x))

    self.x  = self.x.cuda(cuda_device).float()
    self.y = np.loadtxt("{}".format(PATH_y)) -1
    self.augmentation = augmentation
    if kwargs["binary"]:
      self.y = np.where(self.y != 0, 1,0)
    self.y = torch.from_numpy(self.y).cuda(cuda_device).long()
  def get_labels(self):
    return self.y
  def disable_augmentation(self):
    self.augmentation = False
  def enable_augmentation(self,aug):
    self.augmentation = aug

  def __getitem__(self, index):
    x, y = self.x[index], self.y[index]
    if self.augmentation:
      for func in self.augmentation:
        x,y = func(x,y)
    return x,y
  
  def get_n_classes(self):
    print("n classes: {}".format(len(torch.unique(self.y))))
    return len(torch.unique(self.y))
  def get_n_samples(self):
    return self.x.shape[0]
  def get_n_features(self):
    return self.x.shape[1]

  def __len__(self):
    return self.x.shape[0]


class Train_HAR(HAR):

  def __init__(self, augmentation = False,samples_per_class = None,binary = False,one_hot = True,samples_per_epoch = 1,PATH= None,device = None,sub_set_classes = None): 
    super(Train_HAR,self).__init__(True, augmentation,cuda_device = device,samples_per_class = samples_per_class,binary = binary ,one_hot = one_hot,samples_per_epoch = samples_per_epoch,PATH = PATH,sub_set_classes = sub_set_classes)

class Test_HAR(HAR):

  def __init__(self, augmentation = False,samples_per_class = None,binary = False,one_hot = False,samples_per_epoch = 1,device = None,sub_set_classes = None): 
    super(Test_HAR,self).__init__(False ,cuda_device = device, augmentation = augmentation,samples_per_class = samples_per_class,binary = binary ,one_hot = one_hot,samples_per_epoch = samples_per_epoch,sub_set_classes = sub_set_classes)
