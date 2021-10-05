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
from HPO.utils.files import save_obj
  
def compute(hyperparameter,budget = 4, in_model = None , train_dataset = None):
  if train_dataset == None:
    train_dataset = Train_repsol_full(hyperparameter["augmentations"], augmentations = True)


  test_dataset = Test_repsol_full()

  num_classes =  train_dataset.get_n_classes()
  batch_size = 16
  train_dataloader = DataLoader( train_dataset, batch_size=batch_size,
    shuffle = True,drop_last=True,pin_memory=True , collate_fn = collate_fn_padd)


  test_dataloader = DataLoader( test_dataset, batch_size=batch_size, 
                                 drop_last=True,pin_memory=True, collate_fn = collate_fn_padd)
  if in_model == None:
    model = Model(input_size = ( train_dataset.features, ) ,output_size =  num_classes,hyperparameters = hyperparameter)

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
  train_model(model , hyperparameter, train_dataloader , hyperparameter["epochs"], batch_size)
  with torch.no_grad(): #disable back prop to test the model
    model = model.eval()
    correct = 1
    incorrect= 1
    recall_correct = 1
    recall_total = 1
    total = 1 
    for i, (inputs, labels) in enumerate( test_dataloader):
  
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True).view( batch_size ).long().cpu().numpy()
        outputs = model(inputs.float())           
        preds = torch.argmax(outputs.view( batch_size ,2),1).long().cpu().numpy()
        
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
  model_zoo = "/home/snaags/scripts/HPO/src/HPO/model_zoo/"
  torch.save(model.state_dict() , model_zoo+"-Acc-{}-Rec-{}".format(correct/total, recall_correct/recall_total))
  save_obj( hyperparameter , model_zoo+"hps/"+"-Acc-{}-Rec-{}".format(correct/total, recall_correct/recall_total) )
  torch.cuda.empty_cache()
  return [(correct/total), recall_correct/recall_total]
  
  

if __name__ == "__main__":
  dataset = Train_repsol_full(100, augmentations = True)
  while True:

    hyperparameter = {'c1_weight': 4.929843596057779, 'channels': 41, 'layers': 3, 'lr': 0.001, 'normal_cell_1_num_ops': 1, 'normal_cell_1_ops_1_input_1': 0, 'normal_cell_1_ops_1_input_2': 0, 'normal_cell_1_ops_1_type': 'SepConv3', 'normal_cell_1_ops_2_input_1': 1, 'normal_cell_1_ops_2_input_2': 1, 'normal_cell_1_ops_2_type': 'AvgPool5', 'normal_cell_1_ops_3_input_1': 0, 'normal_cell_1_ops_3_input_2': 2, 'normal_cell_1_ops_3_type': 'AvgPool7', 'normal_cell_1_ops_4_input_1': 0, 'normal_cell_1_ops_4_input_2': 1, 'normal_cell_1_ops_4_type': 'SepConv5', 'normal_cell_1_ops_5_input_1': 0, 'normal_cell_1_ops_5_input_2': 2, 'normal_cell_1_ops_5_type': 'Identity', 'normal_cell_1_ops_6_input_1': 0, 'normal_cell_1_ops_6_input_2': 4, 'normal_cell_1_ops_6_type': 'StdConv', 'normal_cell_1_ops_7_input_1': 0, 'normal_cell_1_ops_7_input_2': 4, 'normal_cell_1_ops_7_type': 'Conv5', 'normal_cell_1_ops_8_input_1': 2, 'normal_cell_1_ops_8_input_2': 7, 'normal_cell_1_ops_8_type': 'Conv3', 'normal_cell_1_ops_9_input_1': 5, 'normal_cell_1_ops_9_input_2': 8, 'normal_cell_1_ops_9_type': 'StdConv', 'num_conv': 1, 'num_re': 1, 'p': 0.05, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 'reduction_cell_1_ops_1_type': 'FactorizedReduce', 'window_size': 231}
    hyperparameter = {'c1_weight': 2.120752508120108, 'channels': 41, 'layers': 5, 'lr': 0.001, 'normal_cell_1_num_ops': 9, 'normal_cell_1_ops_1_input_1': 0, 'normal_cell_1_ops_1_input_2': 0, 'normal_cell_1_ops_1_type': 'MaxPool5', 'normal_cell_1_ops_2_input_1': 0, 'normal_cell_1_ops_2_input_2': 0, 'normal_cell_1_ops_2_type': 'StdConv', 'normal_cell_1_ops_3_input_1': 1, 'normal_cell_1_ops_3_input_2': 1, 'normal_cell_1_ops_3_type': 'SepConv5', 'normal_cell_1_ops_4_input_1': 0, 'normal_cell_1_ops_4_input_2': 0, 'normal_cell_1_ops_4_type': 'Conv5', 'normal_cell_1_ops_5_input_1': 0, 'normal_cell_1_ops_5_input_2': 0, 'normal_cell_1_ops_5_type': 'Conv3', 'normal_cell_1_ops_6_input_1': 1, 'normal_cell_1_ops_6_input_2': 2, 'normal_cell_1_ops_6_type': 'MaxPool7', 'normal_cell_1_ops_7_input_1': 5, 'normal_cell_1_ops_7_input_2': 4, 'normal_cell_1_ops_7_type': 'Identity', 'normal_cell_1_ops_8_input_1': 4, 'normal_cell_1_ops_8_input_2': 6, 'normal_cell_1_ops_8_type': 'SepConv7', 'normal_cell_1_ops_9_input_1': 2, 'normal_cell_1_ops_9_input_2': 1, 'normal_cell_1_ops_9_type': 'AvgPool7', 'num_conv': 1, 'num_re': 1, 'p': 0.05, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 'reduction_cell_1_ops_1_type': 'FactorizedReduce', 'window_size': 152} 
    #hyperparameter = {'c1_weight': 1.120752508120108, 'channels': 41, 'layers': 5, 'lr': 0.001, 'normal_cell_1_num_ops': 9, 'normal_cell_1_ops_1_input_1': 0, 'normal_cell_1_ops_1_input_2': 0, 'normal_cell_1_ops_1_type': 'MaxPool5', 'normal_cell_1_ops_2_input_1': 0, 'normal_cell_1_ops_2_input_2': 0, 'normal_cell_1_ops_2_type': 'StdConv', 'normal_cell_1_ops_3_input_1': 1, 'normal_cell_1_ops_3_input_2': 1, 'normal_cell_1_ops_3_type': 'SepConv5', 'normal_cell_1_ops_4_input_1': 0, 'normal_cell_1_ops_4_input_2': 0, 'normal_cell_1_ops_4_type': 'Conv5', 'normal_cell_1_ops_5_input_1': 0, 'normal_cell_1_ops_5_input_2': 0, 'normal_cell_1_ops_5_type': 'Conv3', 'normal_cell_1_ops_6_input_1': 1, 'normal_cell_1_ops_6_input_2': 2, 'normal_cell_1_ops_6_type': 'MaxPool7', 'normal_cell_1_ops_7_input_1': 5, 'normal_cell_1_ops_7_input_2': 4, 'normal_cell_1_ops_7_type': 'Identity', 'normal_cell_1_ops_8_input_1': 4, 'normal_cell_1_ops_8_input_2': 6, 'normal_cell_1_ops_8_type': 'SepConv7', 'normal_cell_1_ops_9_input_1': 2, 'normal_cell_1_ops_9_input_2': 1, 'normal_cell_1_ops_9_type': 'AvgPool7', 'num_conv': 1, 'num_re': 1, 'p': 0.05, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 'reduction_cell_1_ops_1_type': 'FactorizedReduce', 'window_size': 152} 
    compute(hyperparameter , train_dataset = dataset)

