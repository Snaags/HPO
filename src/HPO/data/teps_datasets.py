import numpy as np
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
class TEPS(Dataset):
  def __init__(self, window_size, train , augmentations, samples_per_class = None):
    self.features = 52
    self.n_classes = 21
    if samples_per_class != None:
      self.samples_per_class = [samples_per_class]* self.n_classes
    else:
      self.samples_per_class = [0]* self.n_classes
    path = "/home/snaags/scripts/datasets/TEPS/split/"
    files = os.listdir(path)
    #Get either Training or testing samples
    if train == True:
      filtr = "training"
    else:
      filtr = "testing"
    files_all = [ name for name in files if filtr in name]    
    
    #randomly Subsample to $samples_per_class number of files of each class
    if samples_per_class == None:
      files = files_all
    else:
      files = []
      for i in files_all:
        if i[6:8].isnumeric():
          if self.samples_per_class[int(i[6:8])] > 0:
            files.append(i)
            self.samples_per_class[int(i[6:8])] -=1
        else:
          if self.samples_per_class[int(i[6])] > 0:
            files.append(i)
            self.samples_per_class[int(i[6])] -=1
        if sum(self.samples_per_class) == 0:
          break
    data = []
    self.x_index_address = {}
    self.y_index_address = {}
    self.window = window_size
    for i in files:
      data.append(np.load(path+i))
    self.current_index = 0
    use_all = True
    self.labels = [] 
    for batch in data:
      batch_x = batch[:,1:]
      batch_y = batch[:,0]
      self.add_to_dataset(batch_x,batch_y)
    self.n_features = batch_x.shape[1]
    self.n_samples = self.current_index #* len(self.augmentations)
    self.labels = torch.Tensor(self.labels)
  def add_to_dataset(self,x,y):
    self.x_index_address[self.current_index] = torch.from_numpy(np.swapaxes(x,0,1)).float()
    self.y_index_address[self.current_index] = torch.from_numpy(np.unique(y)).long()
    self.samples_per_class[int(y[0])] += x.shape[0]
    self.labels.append(self.y_index_address[self.current_index])
    self.current_index += 1

  def get_n_samples_per_class(self):
    for i in range(self.n_classes):
      print("Samples in class {}: {}".format(i, self.samples_per_class[i]))    

  def set_window_size(self, window_size):
    self.window = window_size
  def get_labels(self):
    return self.labels
  def __getitem__(self, index):
    #index_address keys are the first usable index in a batch for the window size
    #this means the index

    
    x = self.x_index_address[index]
    y = self.y_index_address[index]


    return x , y
  
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples
  def get_n_features(self):
    return self.n_features

  def __len__(self):
    return self.n_samples -1  


class Train_TEPS(TEPS):

  def __init__(self, window_size = 500, augmentations = False,samples_per_class = None): 
    super().__init__(window_size, True, augmentations,samples_per_class)

class Test_TEPS(TEPS):

  def __init__(self, window_size = 500, augmentations = False,samples_per_class = None): 
    super().__init__(window_size,  False , augmentations,samples_per_class)

