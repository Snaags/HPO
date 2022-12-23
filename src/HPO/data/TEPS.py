import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class TEPS(Dataset):

  def __init__(self, train: bool, augmentation = None, cuda_device = 0):
    PATH = "/home/cmackinnon/scripts/datasets/TEPS/"
    self.cuda_device = cuda_device
    if train:
      data = np.load(PATH+"train_data.npy")
    else:
      data = np.load(PATH+"test_data.npy")
    data = np.swapaxes(data,1,2)
    self.x = torch.from_numpy(data[:,1:,:]).cuda(cuda_device).float()
    self.y = torch.from_numpy(data[:,0,0]).cuda(cuda_device).long()
    self.augmentation = augmentation
  def get_labels(self):
    return self.y

  def __getitem__(self, index):
    x, y = self.x[index], self.y[index]
    if self.augmentation:
      for func in self.augmentation:
        x,y = func(x,y)
    return x,y
  
  def get_n_classes(self):
    return len(torch.unique(self.y))
  def get_n_samples(self):
    return self.x.shape[0]
  def get_n_features(self):
    return self.x.shape[1]

  def __len__(self):
    return self.x.shape[0]


class Train_TEPS(TEPS):

  def __init__(self, window_size = 500, augmentations = False,samples_per_class = None,binary = False,one_hot = True,samples_per_epoch = 1,PATH= None,device = None,sub_set_classes = None): 
    super().__init__(window_size, True, augmentations,samples_per_class,binary = binary ,one_hot = one_hot,samples_per_epoch = samples_per_epoch,PATH = PATH,device = device,sub_set_classes = sub_set_classes)

class Test_TEPS(TEPS):

  def __init__(self, window_size = 500, augmentations = False,samples_per_class = None,binary = False,one_hot = False,samples_per_epoch = 1,device = None,sub_set_classes = None): 
    super().__init__(window_size,  False , augmentations,samples_per_class,binary = binary ,one_hot = one_hot,samples_per_epoch = samples_per_epoch,device = device,sub_set_classes = sub_set_classes)

