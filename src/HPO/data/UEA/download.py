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
  def __init__(self):
    self.initial_path = os.getcwd()
    dataset_dir = input("Enter absolute path to store datasets (e.g. /home/user/scripts/datasets/)")
    os.chdir("")
    self.datasets = {}
    path = ""
    with open("{}UEA_meta.csv".format(path)) as csvfile:
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

  def _get_labels(self, name, _set):
    labels = []
    n_samples = int(self.datasets[name].TestSize if _set == "TEST" else self.datasets[name].TrainSize)
    n_feature = int(self.datasets[name].NumDimensions)
    print(self.datasets[name])
    window_size = int(self.datasets[name].SeriesLength)
    _samples = np.zeros((n_samples, n_feature , window_size))
    for sample in range(n_samples):
        
      labels.append(self.raw_data[0,sample][-1])
      for dim in range(int(self.datasets[name].NumDimensions)):
        _samples[sample,dim,:] = np.asarray(list(self.raw_data[dim,sample])[:-1])
    print(_samples.shape)
    ss = StandardScaler()
    _samples = np.reshape(ss.fit_transform(np.reshape(_samples,(-1, _samples.shape[1]))), _samples.shape)
    return _samples, labels, n_samples , n_feature, window_size

  def _load(self, name, _set):
      if os.path.exists(name):
        os.chdir(str(name))

      self.data = []
      
      for dim in range(1,int(self.datasets[name].NumDimensions)+1):
        self.data.append(list(arff.loadarff(open("{}Dimension{}_{}.arff".format(name , dim ,_set, "rb")))[0]))
      self.raw_data = np.asarray(self.data)
      _dataset = UEA_dataset(*self._get_labels(name ,_set))
      return _dataset
  def load_train(self, name):
      return self._load(name, "TRAIN")

  def load_test(self, name):
      return self._load(name, "TEST")

  def load_all(self, name):
      train,test =  self.load_train(name), self.load_test(name)
      os.chdir(self.initial_path)
      return train, test

class UEA_dataset(Dataset):
  def __init__(self, samples , labels , n_samples , n_features, window_size):
    self.class_names = set(labels)
    self.map = {}
    for idx , name in enumerate(sorted(self.class_names)):
      self.map[name] = idx
    self.x = torch.from_numpy(samples)
    self.y = self.label_encode(labels)
    print(sorted(self.class_names))
    self.n_classes = len(self.class_names)
    self.n_samples = n_samples
    self.n_features = n_features
    self.window_size = window_size
  def __len__(self):
    return self.n_samples
  def __getitem__(self, index):
    return self.x[index] , self.y[index]
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
if __name__ == "__main__":
  datasets = UEA_Handler()
  datasets.get_dataset(name = "BasicMotions")
  x_t, x_T = datasets.load_all(name = "BasicMotions")
  for c, (s,l) in enumerate(x_t):
    print(s.shape, l)
    print("Iterations : {}".format(c))
