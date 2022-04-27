import random
import os 
import numpy as np
import torch
import copy
from HPO.searchspaces.DARTS_config import init_config
from torch.utils.data import Dataset
from HPO.algorithms.algorithm_utils import train_eval
DEBUG_CONST = True
if DEBUG_CONST:
  random.seed(1)
  np.random.seed(1)
  torch.random.manual_seed(1) 

class Labels:
  def __init__(self,clean, classes , samples):
    self.values = clean
    self.is_clean = {x:x in clean for x in range(samples)}
    self.is_predicted = {x:False for x in range(samples)}
    self.is_valid = {x:(self.is_clean[x] or self.is_predicted[x]) for x in range(samples)} #List of indexes with any kind of label
    self.samples = samples
    self.classes = classes
    #self.generate_random()

  def __getitem__(self,index):
    return self.values[index]

  def generate_random(self):
    empty = [ x for x in range(self.samples) if x not in set(list(self.values.keys()))]
    for index in empty:
      self.values[index] = random.choice(list(range(self.classes)))

  def get_valid_samples(self):
    return self.is_valid


class SampleSet(Dataset):
  def __init__(self,mapping):
    self.mapping = mapping
    self.n_samples = len(self.mapping)-1
    self.parent = None 
  def set_main_dataset(self,dataset):
    self.parent = dataset
  def __getitem__(self, index):
    index = self.mapping[index]
    return self.parent[index]
  def __len__(self):
    return self.n_samples
  def get_n_classes(self):
    return self.parent.get_n_classes()
  def get_n_samples(self):
    return self.n_samples
  def get_n_features(self):
    return self.parent.get_n_features()
  def __len__(self):
    return self.n_samples



class Generic(Dataset):
  def __init__(self, x_samples, n_classes , n_features, clean_labels):
    """
    x_samples : array-like of all samples
    clean_labels : diction of confirmed correct labels {index_of_sample : label_value}
    noisy_labels : diction of non confirmed labels {index_of_sample : label_value}

    """
    self.n_features = n_features
    self.n_classes = n_classes
    self.n_samples = len(x_samples)
    self.x_samples = x_samples 
    self.labels = clean_labels
  def __getitem__(self, index):
    x = self.x_samples[index]
    y = self.labels[index]
    return x , y
  
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples
  def get_n_features(self):
    return self.n_features
  def __len__(self):
    return self.n_samples

"""
class SampleSet(Generic):
  def __init__(self,x_samples,n_classes,n_features,labels):
    self.n_samples = x_samples.shape
    super().__init__(x_samples, n_classes , n_features, labels)
"""
class BootSet(Generic):
  def __init__(self, x_samples, n_classes , n_features, clean_labels = None , noisy_labels = None ):
    """
    x_samples : array-like of all samples
    clean_labels : diction of confirmed correct labels {index_of_sample : label_value}
    noisy_labels : diction of non confirmed labels {index_of_sample : label_value}
    """
    self.n_samples = len(x_samples)
    print("Number of samples: {}".format(self.n_samples))
    self.x_samples = x_samples 
    self.labels = Labels(clean_labels, n_classes,self.n_samples)
    self.n_features = n_features
    self.n_classes = n_classes

  def generate_training_sample(self,size):
    mapping,indexs = self.generate_mapping(size)
    return SampleSet(mapping)

  def generate_mapping(self, size):
    is_valid = self.labels.get_valid_samples()
    mapping  = {}
    sub_sample_indexs = []
    samples = list(range(self.n_samples-1))
    index = 0
    while len(mapping) < size:
      sample = samples.pop(random.randint(0,len(samples)-1))
      if is_valid[sample]:
        mapping[index] = sample
        sub_sample_indexs.append(sample)
        index+=1
    return mapping,sub_sample_indexs
      
def split_x_y(files : list, path : str):
    x_samples = {}
    clean_labels = {}
    for index ,i in enumerate(files):
      data = np.reshape(np.load(path+i),(53,-1))
      x = data[1:,:]
      y = data[0,0]
      x_samples[index] = x 
      clean_labels[index]=y
    return x_samples , clean_labels
    

def load_test():

    path = "/home/snaags/scripts/datasets/TEPS/split/"
    files = os.listdir(path)
    filtr = "testing"
    files_test = [ name for name in files if filtr in name]    
    x_validation , y_validation = split_x_y(files_test,path)
    #Random subset of keys and samples 
    
    n_features = 52
    n_classes = 21
    return Generic(x_validation,  n_classes , n_features, y_validation) 

def load_train(num_clean = 10000,total_samples = 10000):
    path = "/home/snaags/scripts/datasets/TEPS/split/"
    files = os.listdir(path)
    filtr = "training"
    files_train = [ name for name in files if filtr in name]    
    x_samples , clean_labels = split_x_y(files_train, path)
    #Random subset of keys and samples 
    if total_samples != None:
      subset_keys = np.random.choice( a = list(x_samples.keys()) , size = total_samples ,replace = False)
      #x_samples = {key:torch.from_numpy(x_samples[key]) for key in subset_keys}
      #clean_labels = {key:torch.from_numpy(clean_labels[key]) for key in subset_keys}
      x_samples = {key:x_samples[key] for key in subset_keys}
      clean_labels = {key:clean_labels[key] for key in subset_keys}
      samples = {}
      labels = {}
      for new_index, (label,sample) in enumerate(zip(clean_labels,x_samples)):
        samples[new_index] = x_samples[sample]
        labels[new_index] = clean_labels[sample]
    #Random subset keys for testing
    clean_labels_subset_keys = np.random.choice( a = list(labels.keys()) , size = num_clean ,replace = False)
    clean_labels_subset = {key:labels[key] for key in clean_labels_subset_keys}

    
    n_features = 52
    n_classes = 21
    return BootSet(samples, n_classes, n_features, clean_labels_subset)

def compute_size():
  #Place holder
  return 500

global test_data
test_data = load_test()
global bootloader
bootloader = load_train()
def main(worker):
    
  #Settings
  N_ITERATIONS = 100
  CORES = 1
  BATCH = 24
  ##Set Up  
  cs = init_config()
  train = train_eval( worker, CORES, filename = "bootloader.csv", handle_dataset = True)
  

  for i in range(N_ITERATIONS):
    ##Iteration
    configs = []
    training_samples = []
    for _ in range(BATCH):
      size = compute_size()
      configs.append( cs.sample_configuration())
      training_samples.append([ bootloader.generate_training_sample(size)])
      
    acc , recall , config = train.eval( population = configs, datasets =training_samples )
    print(acc, recall)
    
if __name__ == "__main__":
  from HPO.workers.boot_worker import compute as worker
  
  main(worker)
  








