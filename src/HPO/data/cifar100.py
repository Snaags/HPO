import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class CIFAR100(Dataset):
  def __init__(self,name,device,augmentation = False):
    self.PATH = "{}{}".format("/home/cmackinnon/scripts/datasets/cifar-100-python/",name)
    self.augmentation = augmentation
    ##LOAD SAMPLES AND LABELS FROM .npy FILE
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    data_train = unpickle(self.PATH)
    #LOAD IMAGES FROM FILE AND RESHAPE TO [SAMPLES, CHANNELS, ROWS, COLUMNS]
    self.x = torch.from_numpy(
        data_train[b'data'].reshape(-1,3,32,32)
        ).cuda(device = device)
    #LOAD LABELS
    self.y = torch.from_numpy(
        np.asarray(data_train[b'fine_labels'] )
        ).cuda(device = device)
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
class CIFAR100_Train(CIFAR100):
  def __init__(self,device,**kwargs):
    name ="train"
    super(CIFAR100_Train,self).__init__(name = name, device = device,**kwargs)
    
class CIFAR100_Test(CIFAR100):
  def __init__(self,device,**kwargs):
    name = "test"
    super(CIFAR100_Test,self).__init__(name = name, device = device,**kwargs)


