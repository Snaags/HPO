import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class UEA(Dataset):
  def __init__(self, name,device,augmentation = False,classes=None):
    self.PATH = "/home/cmackinnon/scripts/datasets/UEA_NPY/"
    self.augmentation = augmentation
    ##LOAD SAMPLES AND LABELS FROM .npy FILE
    x = []
    y = []
    for n in name:
      x.append(np.load("{}{}_samples.npy".format(self.PATH,n)))
      y.append(np.load("{}{}_labels.npy".format(self.PATH,n)))
    self.x = torch.from_numpy(np.concatenate(x,axis = 0)).cuda(device = device).float()
    self.y = torch.from_numpy(np.concatenate(y,axis = 0)).cuda(device = device)
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
    return x,y.long()
  def __len__(self):
    return len(self.y)

  def get_n_classes(self):
    return self.n_classes
    
  def get_n_features(self):
    return self.n_features
class UEA_Train(UEA):
  def __init__(self, name,device,**kwargs):
    name = "{}_{}".format(name,"train")
    super(UEA_Train,self).__init__(name = [name], device = device,**kwargs)
    
class UEA_Test(UEA):
  def __init__(self, name,device,**kwargs):
    name = "{}_{}".format(name,"test")
    super(UEA_Test,self).__init__(name = [name], device = device)

class UEA_Full(UEA):
  def __init__(self, name,device,**kwargs):
    name = ["{}_{}".format(name,"train") , "{}_{}".format(name,"test")]
    super(UEA_Full,self).__init__(name = name, device = device,**kwargs)
