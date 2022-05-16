
import numpy as np
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
from HPO.utils.time_series_augmentation import permutation , magnitude_warp, time_warp, window_slice, jitter, scaling, rotation
import os

class repsol_full(Dataset):

  def __init__(self, path_dir : str, max_size = 3000, min_size = 500):
    self.path = "{}/scripts/datasets/".format(os.environ["HOME"])+path_dir
    data = []
    self.x_index_address = {}
    self.y_index_address = {}
    #index_list sum()
    files = os.listdir(self.path)
    self.device_track = {}
    for c,i in enumerate(files):
      sample = np.reshape(np.load(self.path+i),(-1,28))
      if sample.shape[0] > min_size:
        self.device_track[c] = i[:2]
        data.append(sample[-1*max_size:,:])
    self.current_index = 0
    self.last_batch = []
    self.real_samples = len(data)
    print(self.real_samples)
    for datapoint, batch in enumerate(data):
      batch_x = batch[:,1:]
      batch_y = batch[:,0]
      self.add_to_dataset(batch_x,batch_y)
      self.real_data_index = self.current_index
    self.n_samples = self.current_index #* len(self.augmentations)
    self.features = 27
    self.n_classes = 2

    
  def save_2_file(self, x,y, datapoint , aug_num):
    arr = np.zeros((x.shape[0],x.shape[1]+ 1))
    arr[:,0] = y
    arr[:,1:] = x
    np.save(self.path+"repsol_augmented/{}-{}".format(datapoint, aug_num), arr)
  def get_linked(self, idx):
    return self.aug_source_dict[idx]

  def add_to_dataset(self,x,y):

    self.x_index_address[self.current_index] = torch.from_numpy(np.swapaxes(x, 0,1))
    self.y_index_address[self.current_index] = torch.from_numpy(np.unique(y).reshape((1,)))
    self.current_index += 1 

  def set_window_size(self, window_size):
    self.window = window_size

  def __getitem__(self, index):
    #index_address keys are the first usable index in a batch for the window size
    #this means the index
    x = self.x_index_address[index]
    y = self.y_index_address[index]
    self.last_batch.append(index)   

    return x , y
  
  def y(self):#
    out = []
    for i in self.y_index_address:
      out.append(self.y_index_address[i].numpy())
    return out
  def get_source(self):
    sources = []
    for i in self.last_batch:
      sources.append(self.device_track[i])
    self.last_batch = []
    return sources
  
  def get_id(self):
    return self.last_batch
    

  def get_n_classes(self):
    return self.n_classes
  
  def get_n_features(self):
    return self.features
  def get_n_samples(self):
    return self.n_samples

  def __len__(self):
    return self.n_samples 


class Train_repsol_full(repsol_full):

  def __init__(self, augmentations_num = 200, augmentations = True , sections = [ "1A", "1B", "1C"]): 
    path = "{}/scripts/datasets/repsol_train".format(os.environ["HOME"])
    train_files = os.listdir(path)
    path_dir = "repsol_train/"
    super().__init__(augmentations_num, train_files,path_dir, augmentations , sections)

class Test_repsol_full(repsol_full):

  def __init__(self, augmentations_num = 200, augmentations = False , sections = [ "1A", "1B", "1C" ]): 
    path = "{}/scripts/datasets/repsol_test".format(os.environ["HOME"])
    test_files = os.listdir(path)
    path_dir = "repsol_test/"
    super().__init__(augmentations_num, test_files ,path_dir, augmentations, sections)

class Mixed_repsol_full(repsol_full):
  def __init__(self, path_dir = None):
    path = "{}/scripts/datasets/repsol_mixed".format(os.environ["HOME"])
    if path_dir == None: 
      path_dir = "repsol_mixed/"
    super().__init__( path_dir)

class repsol_unlabeled(Dataset):
  def __init__(self, window_size = 500):
    self.soft_labels = False
    path = "{}/scripts/datasets/repsol_unlabeled/".format(os.environ["HOME"])
    data = []
    self.index_address = {}

    files = os.listdir(path)

    for i in files:
      data.append(np.reshape(np.load(path+i),(-1,27)))
    self.current_index = 0


  
    for batch in data:
      count = window_size
      if batch.shape[0] != 0:
        while batch.shape[0] - count > window_size:
          self.add_to_dataset(batch[count-window_size:count,:])
          count += window_size

      
    
    self.n_samples = self.current_index #* len(self.augmentations)
    self.features = 27
    self.n_classes = 2
  def add_to_dataset(self,x):
    
    self.index_address[self.current_index] = torch.from_numpy(x.reshape(x.shape[1], x.shape[0]))
    self.current_index += 1 

  def add_labels(self , y):
    self.y_index_address = {}
    current_index = 0
    for y_s in y:
      self.y_index_address[current_index] = torch.from_numpy(y_s)
      current_index += 1 
    if current_index != self.current_index:
      raise ValueError("Incorrect number of labels")
    self.soft_labels = True


  def set_window_size(self, window_size):
    self.window = window_size

  def __getitem__(self, index):
    #index_address keys are the first usable index in a batch for the window size
    #this means the index
    if self.soft_labels == False:
      x = self.index_address[index]
      return x 
    else:
      x = self.index_address[index]
      x = self.y_index_address[index]
      return x, y
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples

  def __len__(self):
    return self.n_samples 



class repsol_feature(Dataset):
  def __init__(self,name):
    self.path = "{}/scripts/datasets/repsol_features/".format(os.environ["HOME"])
    self.data = np.load(self.path+name)
      
    self.n_classes = 2
    self.n_samples = self.data.shape[0]
    self.n_features = self.data.shape[1] - 1

  def __getitem__(self, index):
    #index_address keys are the first usable index in a batch for the window size
    #this means the index
    x = self.data[index,1:]
    y = self.data[index,0]
    return x , y
  
  def get_n_classes(self):
    return self.n_classes
  
  def get_n_features(self):
    return self.n_features

  def get_n_samples(self):
    return self.n_samples
  
  def __len__(self):
    return self.n_samples 
    


class Train_repsol_feature(repsol_feature):
  def __init__(self): 
    super().__init__("train_selected.npy")

class Test_repsol_feature(repsol_feature):
  def __init__(self): 
    super().__init__("test_selected.npy")

class Mixed_repsol_feature(repsol_feature):
  def __init__(self): 
    super().__init__("full_selected.npy")

