import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SHAR(Dataset):

  def __init__(self,path, augmentation = None, cuda_device = 0,alt_format = False,**kwargs):
    self.cuda_device = cuda_device
    PATH_x = "{}{}".format(path,"data.npy")
    PATH_y = "{}{}".format(path,"labels.npy")
    PATH_group = "{}{}".format(path,"groups.npy")
    #data = np.swapaxes(data,1,2)
    
    self.x = torch.from_numpy(np.load(PATH_x))
    self.groups = np.load(PATH_group)
    self.x  = self.x.to(cuda_device).float()
    self.y = np.load("{}".format(PATH_y)) -1 
    self.augmentation = augmentation
    if kwargs["binary"]:
      self.y = np.where(self.y != 0, 1,0)
    self.y = torch.from_numpy(self.y).to(cuda_device).long()
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

  def get_proportions(self):
    return 0.2
  def get_n_features(self):
    return self.x.shape[1]
  def get_length(self):
    return self.x.shape[2]

  def __len__(self):
    return self.x.shape[0]


class SHAR_TRAIN(SHAR):
  def __init__(self,path = "/home/cmackinnon/scripts/datasets/SHAR/train/", augmentation = None, cuda_device = 0,**kwargs):
     super(SHAR_TRAIN,self).__init__(path =path,augmentation = augmentation, cuda_device = cuda_device, **kwargs)

class SHAR_TEST(SHAR):
  def __init__(self,path = "/home/cmackinnon/scripts/datasets/SHAR/test/", augmentation = None, cuda_device = 0,**kwargs):
     super(SHAR_TEST,self).__init__(path = path,augmentation = augmentation, cuda_device = cuda_device, **kwargs)

class SHAR_PARTITION(SHAR):
  def __init__(self,path = "/home/cmackinnon/scripts/datasets/SHAR/test/", augmentation = None, cuda_device = 0,**kwargs):
     super(SHAR_TEST,self).__init__(path = path,augmentation = augmentation, cuda_device = cuda_device, **kwargs)

class Full_SHAR(SHAR):
  def __init__(self, augmentation = None, cuda_device = 0,**kwargs):
     super(Full_SHAR,self).__init__(path = "/home/cmackinnon/scripts/datasets/SHAR/",augmentation = augmentation, cuda_device = cuda_device, **kwargs)



"""
class SHAR_TEST(Dataset):

  def __init__(self, augmentation = None, cuda_device = 0,**kwargs):
    PATH = "/home/cmackinnon/scripts/datasets/SHAR/"
    self.cuda_device = cuda_device
    PATH_x = "/home/cmackinnon/scripts/datasets/SHAR/test/data.npy"#UCI HAR Dataset/train/Inertial Signals"
    PATH_y = "/home/cmackinnon/scripts/datasets/SHAR/test/labels.npy"
    PATH_group = "/home/cmackinnon/scripts/datasets/SHAR/test/groups.npy"
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
  def get_length(self):
    return self.x.shape[2]

  def __len__(self):
    return self.x.shape[0]
"""
