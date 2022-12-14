import json
from HPO.utils.model_graph import ModelGraph
from HPO.searchspaces.graph_search_space import GraphConfigSpace
import numpy as np 
from HPO.data.cifar100 import CIFAR100_Train, CIFAR100_Test
import time
import sys
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
from HPO.workers.worker_wrapper import __compute
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def compute(*args, **kwargs):
  __compute(*args, **kwargs, _compute = _compute)


def _compute(hyperparameter,cuda_device, JSON_CONFIG ):
  ### Configuration 
  with open(JSON_CONFIG) as f:
    data = json.load(f)
    SETTINGS = data["WORKER_CONFIG"]
    SAVE_PATH = data["SEARCH_CONFIG"]["PATH"]
  
  if cuda_device == None:
     cuda_device = 3
  torch.cuda.empty_cache()

  torch.cuda.set_device(cuda_device)
  torch.set_float32_matmul_precision('high')
  #torch.autograd.set_detect_anomaly(True)
  #with torch.autograd.profiler.profile() as prof:
  for i in range(1):
      ##Dataset Initialisation
      #datasets = UEA_Handler("/home/cmackinnon/scripts/datasets/UEA/")
      name = SETTINGS["DATASET_CONFIG"]["NAME"]
      #train_args = [False, cuda_device ,None,1]
      # test_args = [False, cuda_device , None,1]
      if "AUGMENTATIONS" in SETTINGS:
        augs = aug.initialise_augmentations(SETTINGS["AUGMENTATIONS"])
      else: 
        augs = None 
      train_dataset = CIFAR100_Train(cuda_device)
      test_dataset = CIFAR100_Test(cuda_device)
      #test_dataset = datasets.load_all(name,train_args,test_args)

      
      print("Cuda Device Value: ", cuda_device)

      n_classes = train_dataset.get_n_classes()
      multibatch = False
      torch.cuda.empty_cache()
      trainloader = torch.utils.data.DataLoader(
                              train_dataset,shuffle = True,
                              batch_size=SETTINGS["BATCH_SIZE"], drop_last = True)
      testloader = torch.utils.data.DataLoader(
                          test_dataset,shuffle = True,
                          batch_size= SETTINGS["BATCH_SIZE"] ,drop_last = True)
      n_classes = test_dataset.get_n_classes()
      evaluator = Evaluator(SETTINGS["BATCH_SIZE"], test_dataset.get_n_classes(),cuda_device,testloader = testloader)   
      print("classes: {}".format(train_dataset.get_n_classes()))
      g = GraphConfigSpace(150)
      s = g.sample_configuration()
      s = s[0]
      model = ModelGraph(train_dataset.get_n_features(),32,train_dataset.get_n_classes(),217,s["graph"],s["ops"],device = cuda_device,data_dim =2)

      model = model.cuda(device = cuda_device)
      model = torch.compile(model)
      #torch._dynamo.config.verbose=True
      #explanation, out_guards, graphs, ops_per_graph,break_reasons, explanation_verbose = torch._dynamo.explain(model, torch.rand(16,3,32,32).cuda(3))
      """
      ### Train the model
      """
      #print(explanation_verbose)
      params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print("Size: {}".format(params))
      train_model(model , SETTINGS, trainloader , cuda_device,logger = False, evaluator = evaluator if SETTINGS["LIVE_EVAL"] else None) 
    #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
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
  torch.save(model.state_dict(),"{}/weights/{:.02f}-{}".format(SAVE_PATH,acc,hyperparameter["ID"]))
  return acc, recall,params


if __name__ == "__main__":
  for i in range(500):
    with open(sys.argv[1]) as f:
      HP = json.load(f)["WORKER_CONFIG"]
      HP["ID"] = i
    _compute(HP,2,sys.argv[1])
