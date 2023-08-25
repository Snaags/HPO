import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
import random 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class UEA(Dataset):
  def __init__(self, name,device,augmentation = False,classes=None, **kwargs):
    if "path" in kwargs and kwargs["path"] != None:
      self.PATH = kwargs["path"]
    else:
      self.PATH = "/home/snaags/scripts/datasets/UEA_NPY/"
    self.augmentation = augmentation
    ##LOAD SAMPLES AND LABELS FROM .npy FILE
    x = []
    y = []
    auxy = []
    self.sizes = []

    for n in name:
      if not os.path.exists("{}{}_samples.npy".format(self.PATH,n)):
        print("WARNING ONE OF THE DATASETS (TRAIN OR TEST DOES NOT EXIST)")
        print("{}{}_samples.npy".format(self.PATH,n))
        continue
      x.append(np.load("{}{}_samples.npy".format(self.PATH,n)))
      x[-1] = np.nan_to_num(x[-1])
      self.sizes.append(x[-1].shape[0])
      y.append(np.load("{}{}_labels.npy".format(self.PATH,n)))
      auxy.append(np.load("{}{}_auxlabels.npy".format(self.PATH,n)))
      
      if "{}_groups.npy".format(n) in os.listdir(self.PATH):
        self.groups = np.load("{}{}_groups.npy".format(self.PATH,n))
    self.x = torch.from_numpy(np.concatenate(x,axis = 0)).to(device = device,dtype = torch.float16,non_blocking = True)
    self.y = np.concatenate(y,axis = 0)
    self.auxy = np.concatenate(auxy,axis = 0)
    if kwargs["binary"]:
      self.y = np.where(self.y != 0, 1,0)
    self.y = torch.from_numpy(self.y).to(non_blocking = True,device = device).long()
    self.auxy = torch.from_numpy(self.auxy).to(non_blocking = True,device = device).long()
    
    if classes != None:
        """
        for c in classes:
            np.where(self.y == c)
        """
    self.n_classes = len(torch.unique(self.y))
    
    self.n_features = self.x.shape[1]
  def __getitem__(self,index):
    x ,y,auxy = self.x[index], self.y[index],self.auxy[index]
    if self.augmentation and self.aug_active:
      for f in self.augmentation:
        x,y,auxy = f(x,y,auxy)
    return x.float(),y.float(),auxy.float()

  def update_device(self,device):
    self.x.to(device)
    self.y.to(device)
  def __len__(self):
    return len(self.y)
  def enable_augmentation(self,augs = None,ids = None):

    self.aug_active = True 
    if augs != None:
      self.augmentation = augs
      for i in self.augmentation:
        i.set_train_ids(ids,self)
  def min_samples_per_class(self):
    unique,counts = np.unique(self.y.cpu().numpy(), return_counts=True)
    return min(counts)
  def disable_augmentation(self):
    self.aug_active = False 
  def get_groups(self, train_id, test_id):
      train_groups = self.groups[train_id]
      test_groups = self.groups[test_id]
      #print("Groups in train are: {} with a total length: {}".format(np.unique(train_groups),len(train_groups)))
      #print("Groups in test are: {} with a total length: {}".format(np.unique(test_groups),len(test_groups)))
  def get_n_classes(self):
    return self.n_classes
  def get_n_features(self):
    return self.n_features
  def get_length(self):
    return self.x.shape[2]

  def get_proportions(self):
    if len(self.sizes) == 1:
      return 0.1
    return self.sizes[1]/ sum(self.sizes)

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

class Train_CharacterTrajectories(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("CharacterTrajectories","train")
    super(Train_CharacterTrajectories,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class Test_CharacterTrajectories(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("CharacterTrajectories","test")
    super(Test_CharacterTrajectories,self).__init__(name = [name], device = cuda_device,**kwargs)


class Train_LSST(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("LSST","train")
    super(Train_LSST,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class Test_LSST(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("LSST","test")
    super(Test_LSST,self).__init__(name = [name], device = cuda_device,**kwargs)

class Full_LSST(UEA):
  def __init__(self, cuda_device,**kwargs):
    name = ["{}_{}".format("LSST","train") , "{}_{}".format("LSST","test")]
    super(Full_LSST,self).__init__(name = name, device = cuda_device,**kwargs)

class Validation_LSST(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("LSST","validation")
    super(Validation_LSST,self).__init__(name = [name], device = cuda_device,**kwargs)
    


class Train_PhonemeSpectra(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("PhonemeSpectra","train")
    super(Train_PhonemeSpectra,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class Test_PhonemeSpectra(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("PhonemeSpectra","test")
    super(Test_PhonemeSpectra,self).__init__(name = [name], device = cuda_device,**kwargs)

class Full_PhonemeSpectra(UEA):
  def __init__(self, cuda_device,**kwargs):
    name = ["{}_{}".format("PhonemeSpectra","train") , "{}_{}".format("PhonemeSpectra","test")]
    super(Full_PhonemeSpectra,self).__init__(name = name, device = cuda_device,**kwargs)
 

class Train_EthanolConcentration(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("EthanolConcentration","train")
    super(Train_EthanolConcentration,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class Test_EthanolConcentration(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("EthanolConcentration","test")
    super(Test_EthanolConcentration,self).__init__(name = [name], device = cuda_device,**kwargs)
class Full_EthanolConcentration(UEA):
  def __init__(self, cuda_device,**kwargs):
    name = ["{}_{}".format("EthanolConcentration","train") , "{}_{}".format("EthanolConcentration","test")]
    super(Full_EthanolConcentration,self).__init__(name = name, device = cuda_device,**kwargs)


    

class Train_PenDigits(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("PenDigits","train")
    super(Train_PenDigits,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class Test_PenDigits(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("PenDigits","test")
    super(Test_PenDigits,self).__init__(name = [name], device = cuda_device,**kwargs)

class Validation_PenDigits(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("PenDigits","validation")
    super(Validation_PenDigits,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class True_Test_PenDigits(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("PenDigits","true_test")
    super(True_Test_PenDigits,self).__init__(name = [name], device = cuda_device,**kwargs)

class Full_PenDigits(UEA):
  def __init__(self, cuda_device,**kwargs):
    name = ["{}_{}".format("PenDigits","train") , "{}_{}".format("PenDigits","test")]
    super(Full_PenDigits,self).__init__(name = name, device = cuda_device,**kwargs)

class Validation_PhonemeSpectra(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("PhonemeSpectra","validation")
    super(Validation_PhonemeSpectra,self).__init__(name = [name], device = cuda_device,**kwargs)


class Train_FaceDetection(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("FaceDetection","train")
    super(Train_FaceDetection,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class Test_FaceDetection(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("FaceDetection","test")
    super(Test_FaceDetection,self).__init__(name = [name], device = cuda_device,**kwargs)

class Full_FaceDetection(UEA):
  def __init__(self, cuda_device,**kwargs):
    name = ["{}_{}".format("FaceDetection","train") , "{}_{}".format("FaceDetection","test")]
    super(Full_FaceDetection,self).__init__(name = name, device = cuda_device,**kwargs)

class Validation_FaceDetection(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("FaceDetection","val")
    super(Validation_FaceDetection,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class True_Test_FaceDetection(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("FaceDetection","true_test")
    super(True_Test_FaceDetection,self).__init__(name = [name], device = cuda_device,**kwargs)




class Train_UWaveGestureLibrary(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("UWaveGestureLibrary","train")
    super(Train_UWaveGestureLibrary,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class Test_UWaveGestureLibrary(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("UWaveGestureLibrary","test")
    super(Test_UWaveGestureLibrary,self).__init__(name = [name], device = cuda_device,**kwargs)

class Full_UWaveGestureLibrary(UEA):
  def __init__(self, cuda_device,**kwargs):
    name = ["{}_{}".format("UWaveGestureLibrary","train") , "{}_{}".format("UWaveGestureLibrary","test")]
    super(Full_UWaveGestureLibrary,self).__init__(name = name, device = cuda_device,**kwargs)
 

class Train_SpokenArabicDigits(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("SpokenArabicDigits","train")
    super(Train_SpokenArabicDigits,self).__init__(name = [name], device = cuda_device,**kwargs)
    
class Test_SpokenArabicDigits(UEA):
  def __init__(self,cuda_device,**kwargs):
    name = "{}_{}".format("SpokenArabicDigits","test")
    super(Test_SpokenArabicDigits,self).__init__(name = [name], device = cuda_device,**kwargs)

class Full_SpokenArabicDigits(UEA):
  def __init__(self, cuda_device,**kwargs):
    name = ["{}_{}".format("SpokenArabicDigits","train") , "{}_{}".format("SpokenArabicDigits","test")]
    super(Full_SpokenArabicDigits,self).__init__(name = name, device = cuda_device,**kwargs)



class Train_N(UEA):
  def __init__(self,ds,cuda_device,**kwargs):
    name = "{}_{}".format(ds,"train")
    super(Train_N,self).__init__(name = [name], device = cuda_device,**kwargs)

class Retrain_N(UEA):
  def __init__(self,ds,cuda_device,**kwargs):
    name = ["{}_{}".format(ds,"train"),"{}_{}".format(ds,"test")]
    super(Retrain_N,self).__init__(name = name, device = cuda_device,**kwargs)
    
class Test_N(UEA):
  def __init__(self,ds,cuda_device,**kwargs):
    name = "{}_{}".format(ds,"test")
    super(Test_N,self).__init__(name = [name], device = cuda_device,**kwargs)


class Full_N(UEA):
  def __init__(self,ds, cuda_device,**kwargs):
    name = ["{}_{}".format(ds,"train") , "{}_{}".format(ds,"test")]
    super(Full_N,self).__init__(name = name, device = cuda_device,**kwargs)

class Validation_N(UEA):
  def __init__(self,ds,cuda_device,**kwargs):
    name = "{}_{}".format(ds,"validation")
    super(Validation_N,self).__init__(name = [name], device = cuda_device,**kwargs)
