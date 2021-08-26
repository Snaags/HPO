import numpy as np
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
from utils.time_series_augmentation import permutation , magnitude_warp, time_warp, window_slice, jitter, scaling, rotation
def none(x):
  return x
class TEPS(Dataset):
  def __init__(self, window_size, x : str , y : str ):
    path = "datasets/TEPS/"
    self.x = torch.from_numpy(np.reshape(np.load(path+x),(-1,52)))
    self.y = torch.from_numpy(np.load(path+y))
    
    self.n_samples = self.x.shape[0]
    self.window = window_size
    self.n_classes = len(np.unique(self.y))

  def set_window_size(self, window_size):
    self.window = window_size
  def __getitem__(self, index):
    while index+self.window > self.n_samples:
      index = random.randint(0,self.n_samples)
    x = self.x[index:index+self.window]
    y = self.y[index+self.window-1]
    x = x.reshape(52,self.window)
    return x, y
  
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples

  def __len__(self):
    return self.n_samples - self.n_samples%self.window


class Train_TEPS(TEPS):

  def __init__(self, window_size = 200): 
    super().__init__(window_size, "x_train.npy","y_train.npy")

class Test_TEPS(TEPS):
  def __init__(self, window_size = 200): 
    super().__init__(window_size, "x_test.npy","y_test.npy")

class BTC(Dataset):
  def __init__(self, window_size, prediction_distance, split):
    path = "data/data/BTC.npy"
    self.prediction_distance = prediction_distance
    self.x = torch.from_numpy(np.reshape(np.load(path),(-1,10)))
    if split > 0.5:
      self.x = self.x[:int(self.x.shape[0]*split)]
    else:
      self.x = self.x[-int(self.x.shape[0]*split):]
    self.n_samples = self.x.shape[0]
    ss = StandardScaler()
    self.x = ss.fit_transform(self.x)
    self.window = window_size
    self.generate_labels()
    self.n_classes = len(np.unique(self.y))

  def generate_labels(self):
    idx = 3
    self.y = []
    for i in range(self.n_samples- self.prediction_distance):
        if self.x[i+self.prediction_distance,idx] > self.x[i,idx]:
          self.y.append(1)
        else:
          self.y.append(0)
    print(len(self.y))
    print(self.x.shape[0])

  def set_window_size(self, window_size):
    self.window = window_size
  def __getitem__(self, index):
    while index+self.window+self.prediction_distance > self.n_samples:
      index = random.randint(0,self.n_samples)
    x = self.x[index:index+self.window]
    y = self.y[index+self.window-1]
    x = x.reshape(10,self.window)
    return x, y
  
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples

  def __len__(self):
    return self.n_samples - self.n_samples%self.window


class Train_BTC(BTC):

  def __init__(self, window_size = 200, split = 0.95 , pred_dist = 1 ): 
    super().__init__(window_size, split = split, prediction_distance = pred_dist)

class Test_BTC(BTC):
  def __init__(self, window_size = 200, split = 0.05, pred_dist = 1): 
    super().__init__(window_size,split = split, prediction_distance = pred_dist)



class repsol(Dataset):
  def __init__(self, window_size, files : list , augmentations):
    path = "/home/cmackinnon/scripts/repsol_np/"
    data = []
    self.x_index_address = {}
    self.y_index_address = {}
    self.window = window_size
    augmentations_per_batch = 2
    #index_list sum()
    if augmentations == True:
      self.non_warp_augmentations =[jitter, scaling, rotation, permutation ]
      self.warp_augmentations = [magnitude_warp, time_warp]
    for i in files:
      data.append(np.reshape(np.load(path+i),(-1,28)))
    self.current_index = 0
    use_all = True

  
    for batch in data:
      batch_x = batch[:,1:]
      batch_y = batch[:,0]
      self.add_to_dataset(batch_x,batch_y)
      if augmentations:
        if use_all == True:
          x = batch_x.reshape(1,batch_x.shape[0],batch_x.shape[1])
          augs = self.warp_augmentations +self.non_warp_augmentations
          for i in range(len(augs)):
              x = augs.pop(random.randint(0,len(augs) -1))(x)
          self.add_to_dataset(x.reshape(*x.shape[1:]),batch_y)
        else:
          for i in range(augmentations_per_batch):
            augmentations_for_current_batch = []
            temp_augmentation_list = self.non_warp_augmentations.copy()
            augmentations_for_current_batch.append(random.choice(self.warp_augmentations))
            x = batch_x.reshape(1,batch_x.shape[0],batch_x.shape[1])
            while len(augmentations_for_current_batch) < 3:
              augmentations_for_current_batch.append(temp_augmentation_list.pop(random.randint(0,len(temp_augmentation_list)-1)))
      
            for augmentation_function in augmentations_for_current_batch:
              x = augmentation_function(x)
            self.add_to_dataset(x.reshape(*x.shape[1:]),batch_y)

    
    self.n_samples = self.current_index #* len(self.augmentations)
    self.features = 27
    self.n_classes = 2
  def add_to_dataset(self,x,y):
    self.x_index_address[self.current_index] = torch.from_numpy(x)
    self.y_index_address[self.current_index] = torch.from_numpy(y)
    self.current_index += x.shape[0] - self.window

  def set_window_size(self, window_size):
    self.window = window_size

  def __getitem__(self, index):
    #index_address keys are the first usable index in a batch for the window size
    #this means the index
    addr = 0
    augmentation_idx = 0
    #while index > self.real_samples:
    #  index -= self.real_samples
    #  augmentation_idx += 1 


    for i in self.x_index_address:
      if (index - i) < 0:
        break
      else:
        addr = i

    index -= addr 
    x = self.x_index_address[addr]
    y = self.y_index_address[addr]

    x = x[index:index+self.window]
    y = y[index+self.window-1]

    return x.reshape( 27,self.window ) , y
  
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples

  def __len__(self):
    return self.n_samples - self.n_samples%self.window


class Train_repsol(repsol):

  def __init__(self, train_files, window_size = 200, augmentations = True): 
    super().__init__(window_size, train_files, augmentations)

class Test_repsol(repsol):

  def __init__(self,test_files, window_size = 200, augmentations = False): 
    super().__init__(window_size, test_files , augmentations)
