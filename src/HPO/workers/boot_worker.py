import numpy as np 
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
from HPO.utils.worker_helper import train_model, collate_fn_padd, train_model_bt, collate_fn_padd_x, train_model_aug
from HPO.utils.weight_freezing import freeze_FCN, freeze_resnet
from HPO.utils.ResNet1d import resnet18
from HPO.utils.files import save_obj
from HPO.workers.repeat_worker import worker_wrapper, one_out_cv_aug, one_out_cv
from queue import Empty
from sklearn.model_selection import KFold
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

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
      acc , rec =  _compute(hyperparameter = config , cuda_device = device, train_data = dataset[0], test_data = dataset[1])
      datasets.task_done()
      res.put([config , acc , rec ]) 

  torch.cuda.empty_cache()
  
def _compute(hyperparameter,budget = 4 , train_data = None,  test_data = None, cuda_device = None,plot_queue = None, model_id = None, binary = False):
  if cuda_device == None:
     cuda_device = 0# torch.cuda.current_device()
  TIMER = time.time()
  torch.cuda.set_device(cuda_device)
  print("Cuda Device Value: ", cuda_device)
  batch_size = 20
  trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  
  n_features = train_data.get_n_features()  
  n_classes = train_data.get_n_classes()
  


  gen = config_space_2_DARTS(hyperparameter)
  acc_full = [0,0]
  recall_full = [0,0]
  model = NetworkMain(n_features,10,num_classes= n_classes , layers = hyperparameter["layers"], auxiliary = False,drop_prob = hyperparameter["p"], genotype = gen, binary = binary)
  model = model.cuda(device = cuda_device)

  print("Training Data Size {} -- Testing Data Size {}".format(len(trainloader), len(testloader)))
  """
  ## Train the model
  """
  train_model(model , hyperparameter, trainloader , 50, batch_size , cuda_device, augment_on = 0, graph = plot_queue, binary = binary) 
  """
  ## Test the model
  """
  with torch.no_grad(): #disable back prop to test the model
    #model = model.eval()
    out_data = []
    batch_size_test = batch_size
    correct = 0
    incorrect= 0
    recall_correct = 0
    recall_total = 0
    total = 0
    for i, (inputs, labels) in enumerate( testloader):
        inputs = inputs.cuda(non_blocking=True, device = cuda_device).float()
        labels = labels.cuda(non_blocking=True, device = cuda_device).view( batch_size_test ).long().cpu().numpy()
        outputs = model(inputs).cuda(device = cuda_device)          
        if binary:
          print(outputs)
          for i in outputs:
            preds = (i > THRESHOLD)
            c = (preds == labels[0]).item()
            print("Reported Result {} -- Output: {} -- Prediction: {} -- Label: {}".format(c, outputs, preds ,labels))
        else:
          preds = torch.argmax(outputs.view(batch_size_test,n_classes),1).long().cpu().numpy()
          c = (preds == labels).sum()

        correct += c 
        t = len(labels)
        total += t
        for l,p in zip(labels, preds):
          if l == 1:
            recall_total += 1
            rt = 1 
            if l == p:
              rc = 1
              recall_correct += 1
            else:
              rc = 0
          else:
            rt = 0
        outputs = outputs.cpu().numpy()
  
  print("Total Correct : {} / {} -- Recall : {} / {}".format(correct,total, recall_correct , recall_total)) 
  print() 
  
  acc = correct/total if total else np.NaN
  recall = recall_correct/recall_total if recall_total else np.NaN
  print("Accuracy: ", "%.4f" % ((acc)*100), "%")
  print("Recall: ", "%.4f" % ((recall)*100), "%")
  model_zoo = "{}/scripts/HPO/src/HPO/model_zoo/".format(os.environ["HOME"])
  torch.save(model.state_dict() , model_zoo+"-Acc-{}-Rec-{}".format(acc, recall))
  save_obj( hyperparameter , model_zoo+"hps/"+"-Acc-{}-Rec-{}".format(acc , recall) )
  acc_full[0] += correct
  acc_full[1] += total
  recall_full[0] += recall_correct
  recall_full[1] += recall_total
  recall = recall_full[0]/recall_full[1]
  acc = acc_full[0]/acc_full[1]
  print("Final Scores -- ACC: {} -- REC: {}".format(acc, recall) )
  print("Total Worker run time: {}".format(time.time() - TIMER))
  return acc, recall

