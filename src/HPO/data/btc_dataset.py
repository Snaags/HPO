import numpy as np
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
"""
class BTC(Dataset):
  def __init__(self, window_size, prediction_distance, split = 5,augmentation = False,samples_per_class = None ,device = None):
    self.path ="{}/scripts/datasets/BTC/BTC.npy".format(os.environ["HOME"])
    self.device = device
    self.prediction_distance = prediction_distance
    self.x = torch.from_numpy(np.reshape(np.load(self.path),(-1,10)))[1000000:,:]
    self.augmentation = augmentation

    if split > 0.5:
      self.x = self.x[:int(self.x.shape[0]*split)]
    else:
      self.x = self.x[-int(self.x.shape[0]*split):]
    ss = StandardScaler()
    self.current_index = 0
    self.x_index_address = {}
    self.y_index_address = {}
    self.x = ss.fit_transform(self.x)
    self.window = window_size
    self.generate_percent_labels()
    self.n_features = self.x.shape[1]
    self.n_classes = len(np.unique(self.y))
    
    if samples_per_class != None:
      self.samples_per_class = [samples_per_class]* self.n_classes
    else:
      self.samples_per_class = [0]* self.n_classes
    
    for i in range(0,self.x.shape[0]-window_size,10):
      self.add_to_dataset(self.x[i:i+500],self.y[i:i+500])
    
    for i in range(self.n_classes):
      print(np.count_nonzero(self.y == i))
    print(len(np.unique(self.y)))
    print(self.samples_per_class)
    self.n_samples = self.current_index
  def generate_labels(self):

    idx = 3
    self.y = []
    for i in range(self.n_samples- self.prediction_distance):
        if self.x[i+self.prediction_distance,idx] > self.x[i,idx]:
          self.y.append(1)
        else:
          self.y.append(0)
    print(self.x.shape[0])

  def add_to_dataset(self,x,y):
    self.x_index_address[self.current_index] = torch.from_numpy(np.swapaxes(x,0,1)).float()
    self.y_index_address[self.current_index] = torch.tensor(y[-1]).long()
    self.samples_per_class[int(y[0])] += 1
    self.current_index += 1
    
    ##DEBUG

  def generate_percent_labels(self, classes = [0.01,0.1,0.25,0.5,0.75,0.9,0.99,1]):
    abs_change = np.diff(self.x[:,3])
    self.class_labels = {}
    per_change = np.divide(abs_change , self.x[1:,3])
    i_last = 0 
    self.y = np.zeros(self.x.shape[0])
    for c,i in enumerate(classes):
      self.class_labels[c] = "{}-{}".format(i_last,i)
      print(np.nanquantile(per_change, i))
      hold = np.where( (per_change > np.nanquantile(per_change,i_last)) & (per_change < np.nanquantile(per_change,i)))
      i_last = i 
      self.y[hold] = c
    
    print(len(self.y))
    print(self.x.shape[0])



  def set_window_size(self, window_size):
    self.window = window_size
  def __getitem__(self, index):
    x = self.x_index_address[index].cuda(device = self.device)
    y = self.y_index_address[index].cuda(device = self.device)
    if self.augmentation:
      for func in self.augmentation:
        x, y = func(x,y)
    return x, y
  
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples
  def get_n_features(self):
    return self.n_features
  def __len__(self):
    return self.n_samples
"""

class BTC(Dataset):
  def __init__(self,window_size,augmentation,device,_set,samples = None):
    self.x = np.load("/home/cmackinnon/scripts/datasets/BTC/train_x.npy").reshape(1,-1)
    self.y = np.load("/home/cmackinnon/scripts/datasets/BTC/train_y.npy")
    self.n_classes = 2
    self.samples = samples
    if samples == None:
      self.n_samples = len(self.x) - window_size
    self.n_samples = samples
    self.augmentation = augmentation
    self.x ,self.y = torch.from_numpy(self.x).cuda(device), torch.from_numpy(self.y).cuda(device)
  def __getitem__(self, index):
    if self.samples != None:
      index = random.randint(0,self.x.shape[-1]-self.window_size)
    x = self.x[:,index:self.window_size+index]
    y = self.y[index+self.window_size]
    if self.augmentation:
      for func in self.augmentation:
        x, y = func(x,y)
    return x, y
  
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples
  def get_n_features(self):
    return self.n_features
  def __len__(self):
    return self.n_samples

class Train(BTC):
  def __init__(self, window_size = 500, augmentation = False, device = None): 
    super().__init__(window_size, split = split, augmentation = augmentation, device =device, _set = "train",window_step = window_step,samples = 10000)

class Test(BTC):
  def __init__(self, window_size = 500, augmentation = False,device = None): 
    super().__init__(window_size, split = split, augmentation = augmentation , device =device, _set = "test")


