import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class UEA(Dataset):
  def __init__(self, name,device,augmentation = False):
    self.PATH = "/home/cmackinnon/scripts/datasets/UEA_NPY/"
    ##LOAD SAMPLES AND LABELS FROM .npy FILE
    x = []
    y = []
    for split in split_name:
      x.append(np.load("{}{}_samples.npy".format(self.PATH,name)))
      y.append(np.load("{}{}_samples.npy".format(self.PATH,name)))
    self.x = torch.from_numpy(np.concatenate(x,axis = 0)).cuda(device = device)
    self.y = torch.from_numpy(np.concatenate(y,axis = 0)).cuda(device = device)

  def __getitem__(self,index):
    if augmentation:
      for f in augmentation:
        x,y = f(x,y)
    return self.x[index], self.y[index]

  def __len__(self):
    len(self.y)


class UEA_Train(UEA):
  def __init__(self, name,device,**kwargs):
    name = "{}_{}".format(name,"train")
    super(UEA_Train).__init__(name = [name], device = device,**kwargs)
    
class UEA_Test(UEA):
  def __init__(self, name,device,**kwargs):
    name = "{}_{}".format(name,"test")
    super(UEA_Test).__init__(name = [name], device = device)

class UEA_Full(UEA):
  def __init__(self, name,device,**kwargs):
    name = ["{}_{}".format(name,"train") , "{}_{}".format(name,"test")]
    super(UEA_Full).__init__(name = name, device = device,**kwargs)
