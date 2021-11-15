import numpy as np 
import time
import os 
import matplotlib.pyplot as plt
from HPO.utils.mlp import MLP
import pandas as pd
import torch
from HPO.data.datasets import Test_repsol_feature , Train_repsol_feature
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import random
from HPO.utils.time_series_augmentation import permutation , magnitude_warp, time_warp
from HPO.utils.time_series_augmentation_torch import jitter, scaling, rotation
from HPO.utils.worker_helper import train_model, collate_fn_padd
from HPO.utils.weight_freezing import freeze_all_cells
from HPO.utils.files import save_obj
from HPO.workers.repeat_worker import worker_wrapper, one_out_cv
from queue import Empty
def compute( ID, configs , gpus , res ):
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
    if device != None:
      print("Starting config with device: {}".format(device))
      acc , rec =  _compute(hyperparameter = config , cuda_device = device)
      res.put([config , acc , rec ]) 

  torch.cuda.empty_cache()
@worker_wrapper
@one_out_cv
def _compute(hyperparameter,budget = 4, in_model = None , train_dataset = None,test_dataset = None, cuda_device = None):
  if cuda_device == None:
     cuda_device =  1#torch.cuda.current_device()
  if train_dataset == None:
    train_dataset = Train_repsol_feature()
  torch.cuda.set_device(cuda_device)
  print("Cuda Device Value: ", cuda_device)
  if test_dataset == None:
    test_dataset = Test_repsol_feature()
  
  num_classes =  train_dataset.get_n_classes()
  batch_size = hyperparameter["batch_size"]
  train_dataloader = DataLoader( train_dataset, batch_size=batch_size,
    shuffle = True,drop_last=True )


  test_dataloader = DataLoader( test_dataset, batch_size=1, 
                                 drop_last=True)
  layer_list = []
  for _ in range(hyperparameter["layers"]):
    layer_list.append(hyperparameter["layer_{}".format(_)])
     
  if in_model == None:
    model = MLP(  train_dataset.get_n_features() ,layer_list ,  num_classes)

  else:
    model = in_model
    model.reset_stem(train_dataset.features)  
    model.reset_fc(2)
    model = freeze_all_cells( model )

  model = model.cuda(device = cuda_device)

  print("dataset length:{}".format(len(train_dataset)))
  """
  ## Train the model
  """
  train_model(model , hyperparameter, train_dataloader , 60, batch_size , cuda_device)
  
  """
  ## Test the model
  """
  
  print("test dataset length:{}".format(len(test_dataset)))
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
        preds = torch.argmax(outputs.view( batch_size ,2),1).long().cpu().numpy()
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
  return acc, recall
  
  

if __name__ == "__main__":
  while True:
    #Scores
    #0.813953488372093,0.7317073170731707
    hyperparameter = {'T_0': 10, 'T_mult': 2, 'batch_size': 11, 'c1_weight': 2.8758919771664937, 'epochs': 93, 'layer_0': 15, 'layer_1': 10, 'layer_2': 10, 'layer_3': 17, 'layers': 1, 'lr': 0.010042140512423458,  'normal_cell_1_ops_2_input_1': 0, 'normal_cell_1_ops_2_input_2': 1, 'normal_cell_1_ops_2_type': 'SepConv3', 'normal_cell_1_ops_3_input_1': 2, 'normal_cell_1_ops_3_input_2': 0, 'normal_cell_1_ops_3_type': 'StdConv', 'normal_cell_1_ops_4_input_1': 2, 'normal_cell_1_ops_4_input_2': 1, 'normal_cell_1_ops_4_type': 'SepConv5', 'normal_cell_1_ops_5_input_1': 4, 'normal_cell_1_ops_5_input_2': 3, 'normal_cell_1_ops_5_type': 'Conv7', 'normal_cell_1_ops_6_input_1': 4, 'normal_cell_1_ops_6_input_2': 3, 'normal_cell_1_ops_6_type': 'MaxPool7', 'normal_cell_1_ops_7_input_1': 3, 'normal_cell_1_ops_7_input_2': 4, 'normal_cell_1_ops_7_type': 'Conv3', 'normal_cell_1_ops_8_input_1': 4, 'normal_cell_1_ops_8_input_2': 3, 'normal_cell_1_ops_8_type': 'SepConv5', 'normal_cell_1_ops_9_input_1': 0, 'normal_cell_1_ops_9_input_2': 8, 'normal_cell_1_ops_9_type': 'SepConv7', 'num_conv': 1, 'num_re': 1, 'p': 0.21641929885000444, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 'reduction_cell_1_ops_1_type': 'FactorizedReduce'}
    _compute(hyperparameter )

