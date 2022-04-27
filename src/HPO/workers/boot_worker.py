import numpy as np 
import sys 
import time
import os 
import matplotlib.pyplot as plt
from HPO.utils.model import NetworkMain
from HPO.utils.DARTS_utils import config_space_2_DARTS
from HPO.utils.FCN import FCN 
import pandas as pd
import torch
from HPO.data.datasets import Test_repsol_full , Mixed_repsol_full, repsol_unlabeled
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
from HPO.utils.time_series_augmentation import permutation , magnitude_warp, time_warp
from HPO.utils.time_series_augmentation_torch import jitter, scaling, rotation
from HPO.utils.worker_train import train_model, collate_fn_padd, train_model_bt, collate_fn_padd_x, train_model_aug
from HPO.utils.weight_freezing import freeze_FCN, freeze_resnet
from HPO.utils.ResNet1d import resnet18
from HPO.utils.files import save_obj
from HPO.workers.repeat_worker import worker_wrapper, one_out_cv_aug, one_out_cv
from queue import Empty
from sklearn.model_selection import KFold
from collections import namedtuple
from HPO.utils.worker_score import Evaluator 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

from HPO.algorithms.boot import test_data
from HPO.algorithms.boot import bootloader 

class LivePlot:
  def __init__(self, data_queue):
    style.use('fivethirtyeight')
    self.max = 0
    self.fig = plt.figure()
    self.ax1 = self.fig.add_subplot(1,1,1) 
    self.loss = []
    self.queue = data_queue
    self.ani = animation.FuncAnimation(self.fig, self.animate, interval = 100)

  def fit(self, y):
    x = np.arange(start = 0 ,stop = len(y))
    trend = np.polyfit(x,y,10)
    trendpoly = np.poly1d(trend)
    return x, trendpoly 
    
  def animate(self,i):
      message = self.queue.get(timeout = 10)

      
      self.ax1.clear()
      self.ax1.semilogy(message, lw = 0.5)
      if len(message) > 100:
        x, poly = self.fit(message)
        self.ax1.semilogy(x, poly(x), lw = 1.5, c = "r")
  def decode(self, message):
      pass      
  def show(self):
      plt.show()

def compute( ID = None, configs=None , gpus=None , res = None  ,  datasets = None):

  device = None
  print("Starting process: {}".format(ID))
  while not configs.empty():
    print(datasets.qsize())
    #try:
    if device == None:
      device = gpus.get(timeout = 10)
    config = configs.get(timeout = 10)
    print("Dataset queue is Empty? {}!".format(datasets.empty()))
    dataset = datasets.get(timeout = 30)
    """
    except Empty:
      if device != None:
        gpus.put(device)
    
    except:
      
      torch.cuda.empty_cache()
      if device != None:
        gpus.put(device)
      return
    """
    if dataset != None:
      augs = config["augmentations"]
      print("Got Dataset!")
    else:
      print("no got dataset :(")

    if config != None:
      augs = config["augmentations"]
      print("Got Configuration!")
    else:
      print("no got config :(")
  
  
    if device != None:
      print("Starting config with device: {}".format(device))
      acc , rec =  _compute(hyperparameter = config , cuda_device = device, train_data = dataset[0])
      res.put([config , acc , rec ]) 

  torch.cuda.empty_cache()
  
def _compute(hyperparameter,budget = 4 , train_data = None, cuda_device = None,plot_queue = None, model_id = None, binary = False):
  #import HPO.algorithms.boot as boot
  train_data.set_main_dataset(bootloader)
  train_data = bootloader
  if cuda_device == None:
     cuda_device = 0# torch.cuda.current_device()
  TIMER = time.time()
  torch.cuda.set_device(cuda_device)
  print("Cuda Device Value: ", cuda_device)
  batch_size = 32
  trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,drop_last = True)
  testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,drop_last = True)
  
  n_features = train_data.get_n_features()
  n_classes = train_data.get_n_classes()
  evaluator = Evaluator(batch_size, n_classes,cuda_device)
  gen = config_space_2_DARTS(hyperparameter)
  acc_full = [0,0]
  recall_full = [0,0]
  model = NetworkMain(n_features,50,num_classes= n_classes , layers = hyperparameter["layers"], auxiliary = False,drop_prob = hyperparameter["p"], genotype = gen, binary = binary)
  model = model.cuda(device = cuda_device)

  print("Training Data Size {} -- Testing Data Size {}".format(len(trainloader), len(testloader)))
  """
  ## Train the model
  """
  train_model_aug(model , hyperparameter, trainloader , 50, batch_size , cuda_device, augment_num = 1, graph = plot_queue, binary = binary) 
  """
  ### Test the model
  """
  evaluator.forward_pass(model, testloader,binary)
  evaluator.predictions(model_is_binary = binary)

  ### Get Metrics
  total = evaluator.T()
  acc  =  evaluator.T_ACC()
  recall = evaluator.TPR(1)
  recall_total = evaluator.P(1)

  print("Accuracy: ", "%.4f" % ((acc)*100), "%")
  print("Recall: ", "%.4f" % ((recall)*100), "%")

  ### Save Model
  def save_model(model, hyperparameter):
    model_zoo = "{}/scripts/model_zoo/".format(os.environ["HOME"])
    torch.save(model.state_dict() , model_zoo+"-Acc-{}-Rec-{}".format(acc, recall))
    save_obj( hyperparameter , model_zoo+"hps/"+"-Acc-{}-Rec-{}".format(acc , recall) )

  save_model(model,hyperparameter)

  print("Final Scores -- ACC: {} -- REC: {}".format(acc, recall))
  return acc, recall

