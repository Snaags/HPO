import json
import numpy as np 
from HPO.data.UEA.download import UEA_Handler
import time
import os 
import matplotlib.pyplot as plt
from HPO.utils.model import NetworkMain
from HPO.utils.DARTS_utils import config_space_2_DARTS
from HPO.utils.FCN import FCN 
import pandas as pd
import torch
from HPO.data.teps_datasets import Train_TEPS , Test_TEPS
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import HPO.utils.augmentation as aug
from HPO.utils.train_utils import collate_fn_padd
from HPO.utils.train import train_model, auto_train_model
from HPO.utils.weight_freezing import freeze_FCN, freeze_resnet
from HPO.utils.ResNet1d import resnet18
from HPO.utils.files import save_obj
from queue import Empty
from sklearn.model_selection import StratifiedKFold as KFold
from collections import namedtuple
from HPO.utils.worker_score import Evaluator 
from HPO.utils.worker_utils import LivePlot
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def compute( ID, configs , gpus , res   , JSON_CONFIG):
  device = None
  config = None
  print("Starting process: {}".format(ID))
  while not configs.empty():
    try:
      if device == None:
        device = gpus.get(timeout = 10)
      config = configs.get(timeout = 10)
    except Empty:
      if device != None:
        gpus.put(device)
      
    except:
      torch.cuda.empty_cache()
      if device != None:
        gpus.put(device)
      return
    if config != None:
      print("Got Configuration!")

    if device != None:
      print("Starting config with device: {}".format(device))
      complete = False
      crashes = 0
      acc , rec =  _compute(hyperparameter = config , cuda_device = device,JSON_CONFIG = JSON_CONFIG)
      while not complete:
        try:
          
          complete = True
        except:
          crashes +=1
          print("Model crash: {} ".format(crashes))
          time.sleep(60)
          if crashes == 2:
            print("Final Crash giving score of zero")
            acc , rec = 0 , 0 
            complete = True
      res.put([config , acc , rec ]) 

  torch.cuda.empty_cache()


def _compute(hyperparameter,cuda_device, JSON_CONFIG ):
  
  ### Configuration 
  with open(JSON_CONFIG) as f:
    SETTINGS = json.load(f)["WORKER_CONFIG"]
  
  if cuda_device == None:
     cuda_device = 1
  torch.cuda.empty_cache()

  torch.cuda.set_device(cuda_device)

  ##Dataset Initialisation
  datasets = UEA_Handler("/home/cmackinnon/scripts/datasets/UEA/")
  name = 'HandMovementDirection'
  train_args = [False, cuda_device ,None,1]
  test_args = [False, cuda_device , None,1]
  dataset_train, test_dataset = datasets.load_all(name,train_args,test_args)


  print("Cuda Device Value: ", cuda_device)
  gen = config_space_2_DARTS(hyperparameter,reduction = True)
  print(gen)

  n_classes = dataset_train.get_n_classes()
  multibatch = False
  torch.cuda.empty_cache()
  trainloader = torch.utils.data.DataLoader(
                          dataset_train,collate_fn = collate_fn_padd,shuffle = True,
                          batch_size=SETTINGS["BATCH_SIZE"], drop_last = True)
  testloader = torch.utils.data.DataLoader(
                      test_dataset,collate_fn = collate_fn_padd,shuffle = True,
                      batch_size= SETTINGS["BATCH_SIZE"] ,drop_last = True)
  n_classes = test_dataset.get_n_classes()
  evaluator = Evaluator(SETTINGS["BATCH_SIZE"], test_dataset.get_n_classes(),cuda_device,testloader = testloader)   

  model = NetworkMain(dataset_train.get_n_features(),2**hyperparameter["channels"], num_classes= dataset_train.n_classes, 
                        layers = hyperparameter["layers"], auxiliary = False,drop_prob = SETTINGS["P"], genotype = gen, binary = SETTINGS["BINARY"])
  model = model.cuda(device = cuda_device)
  """
  ### Train the model
  """
  train_model(model , SETTINGS, trainloader , cuda_device,evaluator = evaluator,logger = False) 
  torch.cuda.empty_cache()
  model.eval()
  evaluator.forward_pass(model, testloader,SETTINGS["BINARY"])
  evaluator.predictions(model_is_binary = SETTINGS["BINARY"] , THRESHOLD = SETTINGS["THRESHOLD"])
  total = evaluator.T()
  acc  =  evaluator.T_ACC()
  recall = evaluator.TPR(1)
  recall_total = evaluator.P(1)
  print("Accuracy: ", "%.4f" % ((acc)*100), "%")
  print("Recall: ", "%.4f" % ((recall)*100), "%")

  return acc, recall

