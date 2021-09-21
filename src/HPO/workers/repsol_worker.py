import numpy as np 
import time
import os 
import matplotlib.pyplot as plt
from HPO.utils.model_constructor import Model
import pandas as pd
import torch
from HPO.data.datasets import Test_repsol , Train_repsol
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import random
from HPO.utils.time_series_augmentation import permutation , magnitude_warp, time_warp
from HPO.utils.time_series_augmentation_torch import jitter, scaling, rotation
from HPO.utils.worker_helper import train_model
from HPO.utils.weight_freezing import freeze_all_cells
  
def compute(hyperparameter,budget = 2, in_model = None):
  DATASET_PATH = "/home/snaags/scripts/datasets/repsol_np"
  TRAIN_SPLIT = 0.7

  train_files = [  ]
  files = os.listdir(DATASET_PATH)
  for i in range(int(len(files)*TRAIN_SPLIT)):
    train_files.append(files.pop(random.randint(0,len(files)-1)))

  train_dataset = Train_repsol(train_files,hyperparameter["window_size"], augmentations = False)
  test_dataset = Test_repsol(files,hyperparameter["window_size"])

  num_classes =  train_dataset.get_n_classes()
  batch_size = 64
  train_dataloader = DataLoader( train_dataset, batch_size=batch_size,
    shuffle = True,drop_last=True,pin_memory=True)

  test_dataloader = DataLoader( test_dataset, batch_size=batch_size, 
                                 drop_last=True,pin_memory=True)
  if in_model == None:
    model = Model(input_size = ( train_dataset.features,  train_dataset.window ) ,output_size =  num_classes,hyperparameters = hyperparameter)
  else:
    model = in_model
    model.reset_stem(train_dataset.features)  
    model.reset_fc(2)
    model = freeze_all_cells( model )

  model = model.cuda()


  
  """
  ## Train the model
  """
  
  ###Training Configuration
  train_model(model , hyperparameter, train_dataloader , budget)
  with torch.no_grad(): #disable back prop to test the model
    model = model.eval()
    correct = 1
    incorrect= 1
    recall_correct = 1
    recall_total = 1
    total = 1 
    for i, (inputs, labels) in enumerate( test_dataloader):
  
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True).long().cpu().numpy()
        outputs = model(inputs.float())           
        preds = torch.argmax(outputs,1).long().cpu().numpy()
        
        correct += (preds == labels).sum()
        total += len(labels)
        for l,p in zip(labels, preds):
          if l == 1:
            recall_total += 1
            if l == p:
              recall_correct += 1
  
  print() 
  print("Accuracy: ", "%.4f" % ((correct/total)*100), "%")
  print("Recall: ", "%.4f" % ((recall_correct/recall_total)*100), "%")
  
  torch.cuda.empty_cache()
  return (correct/total) 
  
  

if __name__ == "__main__":
  hyperparameter = {'channels': 27, 'layers': 4, 'lr': 0.0001, 'normal_cell_1_num_ops': 1, 'normal_cell_1_ops_1_input_1': 0, 'normal_cell_1_ops_1_input_2': 0, 'normal_cell_1_ops_1_type': 'Conv3', 'normal_cell_1_ops_2_input_1': 0, 'normal_cell_1_ops_2_input_2': 1, 'normal_cell_1_ops_2_type': 'Identity', 'normal_cell_1_ops_3_input_1': 0, 'normal_cell_1_ops_3_input_2': 2, 'normal_cell_1_ops_3_type': 'SepConv7', 'normal_cell_1_ops_4_input_1': 0, 'normal_cell_1_ops_4_input_2': 1, 'normal_cell_1_ops_4_type': 'AvgPool5', 'normal_cell_1_ops_5_input_1': 2, 'normal_cell_1_ops_5_input_2': 4, 'normal_cell_1_ops_5_type': 'MaxPool7', 'normal_cell_1_ops_6_input_1': 1, 'normal_cell_1_ops_6_input_2': 1, 'normal_cell_1_ops_6_type': 'AvgPool7', 'normal_cell_1_ops_7_input_1': 5, 'normal_cell_1_ops_7_input_2': 6, 'normal_cell_1_ops_7_type': 'AvgPool5', 'normal_cell_1_ops_8_input_1': 2, 'normal_cell_1_ops_8_input_2': 3, 'normal_cell_1_ops_8_type': 'Identity', 'normal_cell_1_ops_9_input_1': 6, 'normal_cell_1_ops_9_input_2': 1, 'normal_cell_1_ops_9_type': 'MaxPool5', 'num_conv': 1, 'num_re': 1, 'p': 0.05, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 'reduction_cell_1_ops_1_type': 'FactorizedReduce', 'window_size': 354} 
  worker = repsol_worker()
  worker.compute(hyperparameter)

