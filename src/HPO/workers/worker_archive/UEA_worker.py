import numpy as np 
import time
import os 
import matplotlib.pyplot as plt
from HPO.utils.model_constructor import Model
import pandas as pd
import torch
from HPO.data.datasets import Test_repsol_full , Train_repsol_full
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import random
from HPO.utils.time_series_augmentation import permutation , magnitude_warp, time_warp
from HPO.utils.time_series_augmentation_torch import jitter, scaling, rotation
from HPO.utils.worker_helper import train_model, collate_fn_padd
from HPO.utils.weight_freezing import freeze_all_cells
from HPO.utils.FCN import FCN 
from HPO.utils.model import NetworkMain
from HPO.utils.DARTS_utils import config_space_2_DARTS
from HPO.data.UEA.download import UEA_Handler
from HPO.utils.files import save_obj
from HPO.workers.repeat_worker import worker_wrapper, one_out_cv_aug, one_out_cv
from queue import Empty
def compute( ID = None, configs=None , gpus=None , res = None  , config = None):

  device = None
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
      augs = config["augmentations"]
      print("Got Configuration!")
    else:
      augs = 0
  

  
    if device != None:
      print("Starting config with device: {}".format(device))
      acc , rec =  _compute(hyperparameter = config , cuda_device = device)
      res.put([config , acc , rec ]) 

  torch.cuda.empty_cache()
def _compute(hyperparameter,budget = 4, in_model = None , train_dataset = None,  test_dataset = None, cuda_device = None):
  if cuda_device == None:
     cuda_device = 1# torch.cuda.current_device()
  torch.cuda.set_device(cuda_device)
  print("Cuda Device Value: ", cuda_device)
  gen = config_space_2_DARTS(hyperparameter)
  uea = UEA_Handler()
  ds_list = uea.list_datasets()
  print(ds_list)
  model = FCN(10)
  model = model.cuda(device = cuda_device)
  for i in ds_list:
    print(i)
    uea.get_dataset(i)
    train_dataset, test_dataset = uea.load_all(i)
    num_classes =  train_dataset.get_n_classes()
    batch_size = 8
    model.reset(train_dataset.get_n_features(),classes = train_dataset.get_n_classes())
    model = model.cuda(device = cuda_device)
    train_dataloader = DataLoader( train_dataset, batch_size=batch_size,
      shuffle = True,drop_last=True)
    test_dataloader = DataLoader( test_dataset, batch_size=1, 
                                   drop_last=False)
    #model = NetworkMain(train_dataset.get_n_features(),hyperparameter["channels"],num_classes= train_dataset.get_n_classes() , layers = hyperparameter["layers"], auxiliary = False,drop_prob = hyperparameter["p"], genotype = gen)
  
    
    print("Training Data Size {} -- Testing Data Size {}".format(len(train_dataset), len(test_dataset)))
    """
    ## Train the model
    """
    train_model(model , hyperparameter, train_dataloader , hyperparameter["epochs"], batch_size , cuda_device)
    
    """
    ## Test the model
    """
    
    with torch.no_grad(): #disable back prop to test the model
      #model = model.eval()
  
      batch_size = 1
      correct = 0
      incorrect= 0
      recall_correct = 0
      recall_total = 0
      total = 0
      for i, (inputs, labels) in enumerate( test_dataloader):
          inputs = inputs.cuda(non_blocking=True, device = cuda_device).float()
          labels = labels.cuda(non_blocking=True, device = cuda_device).view( batch_size ).long().cpu().numpy()
          outputs = model(inputs).cuda(device = cuda_device)          
          preds = torch.argmax(outputs.view(batch_size,test_dataset.get_n_classes()),1).long().cpu().numpy()
          correct += (preds == labels).sum()
          total += len(labels)
          for l,p in zip(labels, preds):
            if l == 1:
              recall_total += 1
              if l == p:
                recall_correct += 1
    print("Total Correct : {} / {} -- Recall : {} / {}".format(correct,total, recall_correct , recall_total)) 
    print() 
    acc = correct/total if total else np.NaN
    recall = recall_correct/recall_total if recall_total else np.NaN
    print("Accuracy: ", "%.4f" % ((acc)*100), "%")
    print("Recall: ", "%.4f" % ((recall)*100), "%")
    model_zoo = "{}/scripts/HPO/src/HPO/model_zoo/".format(os.environ["HOME"])
    torch.save(model.state_dict() , model_zoo+"-Acc-{}-Rec-{}".format(acc, recall))
    save_obj( hyperparameter , model_zoo+"hps/"+"-Acc-{}-Rec-{}".format(acc , recall) )
  

if __name__ == "__main__":
  while True:


    hyperparameter = {'T_0': 3, 'T_mult': 2, 'augmentations': 13, 'batch_size': 8, 'channels': 64, 'epochs': 60, 'layers': 7, 'lr': 0.002039759028016279, 'normal_index_0_0': 1, 'normal_index_0_1': 0, 'normal_index_1_0': 0, 'normal_index_1_1': 1, 'normal_index_2_0': 1, 'normal_index_2_1': 3, 'normal_index_3_0': 2, 'normal_index_3_1': 4, 'normal_node_0_0': 'avg_pool_3x3', 'normal_node_0_1': 'skip_connect', 'normal_node_1_0': 'dil_conv_3x3', 'normal_node_1_1': 'dil_conv_5x5', 'normal_node_2_0': 'dil_conv_5x5', 'normal_node_2_1': 'dil_conv_3x3', 'normal_node_3_0': 'avg_pool_3x3', 'normal_node_3_1': 'skip_connect', 'p': 0.01237919046653286, 'reduction_index_0_0': 0, 'reduction_index_0_1': 0, 'reduction_index_1_0': 2, 'reduction_index_1_1': 2, 'reduction_index_2_0': 3, 'reduction_index_2_1': 2, 'reduction_index_3_0': 4, 'reduction_index_3_1': 4, 'reduction_node_0_0': 'max_pool_3x3', 'reduction_node_0_1': 'max_pool_3x3', 'reduction_node_1_0': 'dil_conv_5x5', 'reduction_node_1_1': 'avg_pool_3x3', 'reduction_node_2_0': 'sep_conv_3x3', 'reduction_node_2_1': 'avg_pool_3x3', 'reduction_node_3_0': 'none', 'reduction_node_3_1': 'avg_pool_3x3'}

    _compute(hyperparameter)

