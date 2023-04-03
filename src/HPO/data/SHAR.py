import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SHAR(Dataset):

  def __init__(self, augmentation = None, cuda_device = 0,**kwargs):
    PATH = "/home/cmackinnon/scripts/datasets/WHAR/"
    self.cuda_device = cuda_device
    PATH_x = "/home/cmackinnon/scripts/datasets/SHAR/data.npy"#UCI HAR Dataset/train/Inertial Signals"
    PATH_y = "/home/cmackinnon/scripts/datasets/SHAR/labels.npy"
    PATH_group = "/home/cmackinnon/scripts/datasets/SHAR/groups.npy"
    #data = np.swapaxes(data,1,2)
    
    self.x = torch.from_numpy(np.load(PATH_x))
    self.groups = np.load(PATH_group)
    self.x  = self.x.cuda(cuda_device).float()
    self.y = np.load("{}".format(PATH_y)) -1 
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

  def get_groups(self, train_id, test_id):
      train_groups = self.groups[train_id]
      test_groups = self.groups[test_id]
      print("Groups in train are: {} with a total length: {}".format(np.unique(train_groups),len(train_groups)))
      print("Groups in test are: {} with a total length: {}".format(np.unique(test_groups),len(test_groups)))
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

