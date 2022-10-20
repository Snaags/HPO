import csv
import torch
import io
import requests
import collections
import zipfile
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import os
from torch.utils.data import Dataset
#import arff
import numpy as np
URL = "http://www.timeseriesclassification.com/Downloads/"


class UEA_Handler:
  def __init__(self,dataset_dir):
    self.initial_path = os.getcwd()
    #os.chdir("")
    self.datasets = {}
    path = dataset_dir
    self.raw_data = {"TRAIN": [] , "TEST":[]}
    self.path = path
    print("Loading Metadata from {}UEA_meta.csv".format(path))
    with open("{}UEA_meta.csv".format(path),"r") as csvfile:
      reader = csv.reader(csvfile,delimiter = ",")
      for c, row in enumerate(reader):
        if c == 0:
          HEADINGS = [x for x in row if len(x)]
          Dataset = collections.namedtuple("Dataset", HEADINGS)
        else:
          self.datasets[row[0]] = Dataset(*row[:9],[x for x in row[10:] if len(x)])
  def list_datasets(self):
    return list(self.datasets.keys())

  def get_dataset(self, name):
      if os.path.exists(name):
        os.chdir(str(name))
      else:
        r = requests.get("{}{}.zip".format(URL,name))
        os.system("mkdir {}".format(name))

        os.chdir(str(name))
        with open("{}.zip".format(name), "wb") as file:
          file.write(r.content)
        os.system("unzip {}.zip".format(name))
        print(os.getcwd())

  def _get_labels(self, name, _set):
    ##Lol I fucked this stuff up, mid conversion to allowing the train and test set to allow arbitrary splits
    labels = []
    if _set == "TEST":
        n_samples = int(self.datasets[name].TestSize)
        data = self.raw_data[_set]
    elif _set == "TRAIN":
        n_samples = int(self.datasets[name].TrainSize)
        data = self.raw_data[_set]
    else:
        n_samples = int(self.datasets[name].TrainSize) + int(self.datasets[name].TestSize)
        print("array 1 shape Shape: {} {}".format(self.raw_data["TRAIN"].shape,self.raw_data["TEST"].shape))
        data = np.concatenate((self.raw_data["TRAIN"],self.raw_data["TEST"]),axis = 1)
        print("Mixed Dataset Shape: {}".format(data.shape))
    
    n_feature = int(self.datasets[name].NumDimensions)
    
    print("Loading: {} ".format(self.datasets[name] ))
    window_size = int(self.datasets[name].SeriesLength)
    _samples = np.zeros((n_samples, n_feature , window_size))
    
    for sample in range(n_samples):
      labels.append(  data[0,sample][-1])
      for dim in range(int(self.datasets[name].NumDimensions)):
        _samples[sample,dim,:] = np.asarray(list(data[dim,sample])[:-1])
    print(_samples.shape)
    ss = StandardScaler()
    _samples = np.reshape(ss.fit_transform(np.reshape(_samples,(-1, _samples.shape[1]))), _samples.shape)
    return _samples, labels, n_samples , n_feature, window_size

  def _load(self, name, _set,args):
        
      self.data = []
                      
      for dim in range(1,int(self.datasets[name].NumDimensions)+1):
        self.data.append(list(arff.loadarff(open("{}{}/{}Dimension{}_{}.arff".format(self.path,name,name , dim ,_set, "rb")))[0]))
      self.raw_data[_set] = np.asarray(self.data)
      print("Name: {} Shape: {}".format(name,self.raw_data[_set].shape))
    
  def load_train(self, name,train_args):
      self._load(name, "TRAIN",train_args)
  def load_test(self, name,test_args):
      self._load(name, "TEST",test_args)

    

  def load_all(self, name, train_args,test_args):
      self.load_train(name,train_args), self.load_test(name,test_args)
      test_dataset = UEA_dataset(*self._get_labels(name ,"TEST"),*test_args)
      train_dataset = UEA_dataset(*self._get_labels(name ,"TRAIN"),*train_args)
      os.chdir(self.initial_path)
      return train_dataset, test_dataset
    
  def load_mixed(self, name, train_args,test_args):
      self.load_train(name,train_args)
      self.load_test(name,test_args)
      _dataset = UEA_dataset(*self._get_labels(name ,"MIXED"),*train_args)
      os.chdir(self.initial_path)
      return _dataset



class UEA_dataset(Dataset):
  def __init__(self, samples , labels , n_samples , n_features, window_size,one_hot = True,device = None, augmentations = None,samples_per_epoch = 1):
    self.class_names = set(labels)
    self.device = device
    self.samples_per_epoch = samples_per_epoch
    self.one_hot = one_hot
    self.augmentations = augmentations
    self.map = {}
    for idx , name in enumerate(sorted(self.class_names)):
      self.map[name] = idx
    self.x = torch.from_numpy(samples).cuda(device = self.device )
    self.y = self.label_encode(labels).cuda(device = self.device )
    print(sorted(self.class_names))
    self.n_classes = len(self.class_names)
    self.n_samples = n_samples * samples_per_epoch
    self.true_n_samples = n_samples
    self.n_features = n_features
    self.window_size = window_size
  def __len__(self):
    return self.n_samples
  def __getitem__(self, index):
    if self.samples_per_epoch > 1:
      if index >= self.true_n_samples-2:
        index = index % (self.true_n_samples-2)
        
    x,y = self.x[index] , self.y[index]

    if self.augmentations:
      for func in self.augmentations:
        x,y = func(x,y)
    if self.one_hot == False:
      return x , y.long()
    return x , y

  def label_encode(self, labels):
    out = []
    for i in labels:
      out.append(self.map[i])
    return torch.Tensor(out)
  
  def get_n_samples(self):
    return self.n_samples

  def get_n_features(self):
    return self.n_features

  def get_n_classes(self):
    return self.n_classes



class TEPS(Dataset):
  def __init__(self, window_size, train , augmentations, samples_per_class = None,binary = False,one_hot = True,samples_per_epoch = 1,PATH = None,device = None):
    self.features = 52
    self.device = device
    self.one_hot = one_hot
    self.augmentations = augmentations
    if binary == True:
      self.n_classes = 2
    else:
      self.n_classes = 21

    if samples_per_class != None and binary == True:
      self.samples_per_class = [int(samples_per_class/20)]*20
      self.samples_per_class = round_average(self.samples_per_class, samples_per_class)
      self.samples_per_class = [samples_per_class] + self.samples_per_class
    elif samples_per_class != None:
      self.samples_per_class = [samples_per_class]* self.n_classes
    else:
      self.samples_per_class = [0]* self.n_classes
    if PATH == None:
      path = "{}/scripts/datasets/TEPS/split/".format(os.environ["HOME"])
    else:
      path = PATH
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
      random.shuffle(files_all)
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
    self.true_labels = [] 
    for batch in data:
      batch_x = batch[:,1:]
      self.true_labels.append(batch[0,0])
      if binary == True and batch[0,0] > 0:
        batch_y = [1]
      else:
        batch_y = batch[:,0]
      self.add_to_dataset(batch_x,batch_y)
    self.n_features = batch_x.shape[1]
    self.n_samples = self.current_index + 1 #* len(self.augmentations)
    self.samples_per_epoch = samples_per_epoch
    if samples_per_epoch > 1:
      self.true_n_samples = self.n_samples
      self.n_samples *= samples_per_epoch
    self.labels = torch.Tensor(self.labels)
  def add_to_dataset(self,x,y):
    if self.one_hot == True:
      self.x_index_address[self.current_index] = torch.from_numpy(np.swapaxes(x,0,1)).float().cuda(device = self.device )
      self.y_index_address[self.current_index] = F.one_hot(torch.from_numpy(np.unique(y)).long(),num_classes = self.n_classes).cuda(device = self.device).long()
    else:
      self.x_index_address[self.current_index] = torch.from_numpy(np.swapaxes(x,0,1)).float().cuda(device = self.device )
      self.y_index_address[self.current_index] = torch.from_numpy(np.unique(y)).cuda(device = self.device).long()
    self.samples_per_class[int(y[0])] += x.shape[0]
    self.current_index += 1
    self.labels.append(y[0])

  def get_n_samples_per_class(self):
    for i in range(self.n_classes):
      print("Samples in class {}: {}".format(i, self.samples_per_class[i]))    

  def set_window_size(self, window_size):
    self.window = window_size
  def get_labels(self):
    return self.labels
  def get_true_labels(self):
    return self.true_labels
  def disable_augmentation(self):
    self.augmentations = False

  def update_augmentation(self, augmentations):
    self.augmentations = augmentations
  def __getitem__(self, index):
    #index_address keys are the first usable index in a batch for the window size
    #this means the index
    if self.samples_per_epoch > 1:
      if index >= self.true_n_samples-2:
        index = index % (self.true_n_samples-2)
    x = self.x_index_address[index]
    y = self.y_index_address[index]
    if self.augmentations:
      for func in self.augmentations:
        x,y = func(x,y)
    if self.one_hot == False:
      return x , y.long()
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

  def __init__(self, window_size = 500, augmentations = False,samples_per_class = None,binary = False,one_hot = True,samples_per_epoch = 1,PATH= None,device = None): 
    super().__init__(window_size, True, augmentations,samples_per_class,binary = binary ,one_hot = one_hot,samples_per_epoch = samples_per_epoch,PATH = PATH,device = device)

class Test_TEPS(TEPS):

  def __init__(self, window_size = 500, augmentations = False,samples_per_class = None,binary = False,one_hot = False,samples_per_epoch = 1,device = None): 
    super().__init__(window_size,  False , augmentations,samples_per_class,binary = binary ,one_hot = one_hot,samples_per_epoch = samples_per_epoch,device = device)



if __name__ == "__main__":
  datasets = UEA_Handler()
  datasets.get_dataset(name = "BasicMotions")
  x_t, x_T = datasets.load_all(name = "BasicMotions")
  for c, (s,l) in enumerate(x_t):
    print(s.shape, l)
    print("Iterations : {}".format(c))
