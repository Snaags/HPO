import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class DWrap(Dataset):
  def __init__(self, name,device,augmentation = False,classes=None, **kwargs):
    if "path" in kwargs and kwargs["path"] != None:
      self.PATH = kwargs["path"]
    else:
      self.PATH = "/home/cmackinnon/scripts/datasets/DWrap_NPY/"
    self.augmentation = augmentation
    ##LOAD SAMPLES AND LABELS FROM .npy FILE
    x = []
    y = []
    self.sizes = []

    for n in name:
      x.append(np.load("{}{}_samples.npy".format(self.PATH,n)))
      self.sizes.append(x[-1].shape[0])
      if n.split("_")[-1] != "test":
        y.append(np.load("{}{}_labels.npy".format(self.PATH,n)))
      
      if "{}_groups.npy".format(n) in os.listdir(self.PATH):
        self.groups = np.load("{}{}_groups.npy".format(self.PATH,n))
    self.x = torch.from_numpy(np.concatenate(x,axis = 0)).to(device = device).float()
    self.x = torch.nan_to_num(self.x)
    self.y = np.concatenate(y,axis = 0)
    if kwargs["binary"]:
      self.y = np.where(self.y != 0, 1,0)
    self.y = torch.from_numpy(self.y).to(device).long()
    
    if classes != None:
        """
        for c in classes:
            np.where(self.y == c)
        """
    self.n_classes = len(torch.unique(self.y))
    
    self.n_features = self.x.shape[1]
  def __getitem__(self,index):
    x ,y = self.x[index], self.y[index]
    if self.augmentation:
      for f in self.augmentation:
        x,y = f(x,y)
    return x,y

  def update_device(self,device):
    self.x.to(device)
    self.y.to(device)
  def __len__(self):
    return len(self.y)
  def enable_augmentation(self,augs):
    self.augmentation = augs
  def min_samples_per_class(self):
    unique,counts = np.unique(self.y.cpu().numpy(), return_counts=True)
    return min(counts)
  def disable_augmentation(self):
    self.augmentation = False 
  def get_groups(self, train_id, test_id):
      train_groups = self.groups[train_id]
      test_groups = self.groups[test_id]
      #print("Groups in train are: {} with a total length: {}".format(np.unique(train_groups),len(train_groups)))
      #print("Groups in test are: {} with a total length: {}".format(np.unique(test_groups),len(test_groups)))
  def get_n_classes(self):
    return self.n_classes
  def get_n_features(self):
    return self.n_features
  def get_length(self):
    return self.x.shape[2]

  def get_proportions(self):
    return self.sizes[1]/ sum(self.sizes)


class Train_N(DWrap):
  def __init__(self,ds,cuda_device,**kwargs):
    name = "{}_{}".format(ds,"train")
    super(Train_N,self).__init__(name = [name], device = cuda_device,**kwargs)


class Test_N(DWrap):
  def __init__(self,ds,cuda_device,**kwargs):
    name = "{}_{}".format(ds,"test")
    super(Test_N,self).__init__(name = [name], device = cuda_device,**kwargs)

class Full_N(DWrap):
  def __init__(self,ds, cuda_device,**kwargs):
    name = ["{}_{}".format(ds,"train") , "{}_{}".format(ds,"test")]
    super(Full_N,self).__init__(name = name, device = cuda_device,**kwargs)

class Validation_N(DWrap):
  def __init__(self,ds,cuda_device,**kwargs):
    name = "{}_{}".format(ds,"validation")
    super(Validation_N,self).__init__(name = [name], device = cuda_device,**kwargs)