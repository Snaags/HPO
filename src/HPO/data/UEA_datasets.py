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
      self.PATH = "/home/cmackinnon/scripts/datasets/UEA_NPY/"
    self.augmentation = augmentation
    ##LOAD SAMPLES AND LABELS FROM .npy FILE
    x = []
    y = []
    self.sizes = []

    for n in name:
      x.append(np.load("{}{}_samples.npy".format(self.PATH,n)))
      self.sizes.append(x[-1].shape[0])
      y.append(np.load("{}{}_labels.npy".format(self.PATH,n)))
    self.x = torch.from_numpy(np.concatenate(x,axis = 0)).to(device = device).float()
    self.x = torch.nan_to_num(self.x)
    self.y = np.concatenate(y,axis = 0)
    if kwargs["binary"]:
      self.y = np.where(self.y != 0, 1,0)
    self.y = torch.from_numpy(self.y).to(device).long()
    print("Length of {}: {}".format(name, self.x.shape[0]))
    if classes != None:
        """
        for c in classes:
            np.where(self.y == c)
        """
    self.n_classes = len(torch.unique(self.y))
    print("number of classes: ",len(torch.unique(self.y)))
    self.n_features = self.x.shape[1]
  def __getitem__(self,index):
    x ,y = self.x[index], self.y[index]
    if self.augmentation:
      for f in self.augmentation:
        x,y = f(x,y)
    return x,y
  def __len__(self):
    return len(self.y)
  def enable_augmentation(self,augs):
    self.augmentation = augs
  def disable_augmentation(self):
    self.augmentation = False 

  def get_n_classes(self):
    return self.n_classes
    
  def get_n_features(self):
    return self.n_features
  def get_length(self):
    return self.x.shape[2]

  def get_proportions(self):
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
 