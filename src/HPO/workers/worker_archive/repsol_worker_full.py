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
    if train_dataset == None:
      train_dataset = Train_repsol_full(hyperparameter["augmentations"], augmentations = True)
    torch.cuda.set_device(cuda_device)
    print("Cuda Device Value: ", cuda_device)
    if test_dataset == None:
      test_dataset = Test_repsol_full()
  
    num_classes =  train_dataset.get_n_classes()
    batch_size = hyperparameter["batch_size"]
    train_dataloader = DataLoader( train_dataset, batch_size=batch_size,
      shuffle = True,drop_last=True , collate_fn = collate_fn_padd)
  
  
    test_dataloader = DataLoader( test_dataset, batch_size=1, 
                                   drop_last=False, collate_fn = collate_fn_padd)
    if in_model == None:
      model = Model(input_size = ( train_dataset.get_n_features(), ) ,output_size =  num_classes,hyperparameters = hyperparameter)
  
    else:
      model = in_model
      model.reset_stem(train_dataset.features)  
      model.reset_fc(2)
      model = freeze_all_cells( model )
  
    model = model.cuda(device = cuda_device)
  
    
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
          print("Label Values: {}".format(labels))
          inputs = inputs.cuda(non_blocking=True, device = cuda_device).float()
          labels = labels.cuda(non_blocking=True, device = cuda_device).view( batch_size ).long().cpu().numpy()
          outputs = model(inputs).cuda(device = cuda_device)          
          print(outputs) 
          preds = torch.argmax(outputs.view(batch_size,2),1).long().cpu().numpy()
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

    hyperparameter = {'lr_step': 2, 'normal_cell_1_ops_8_input_1': 0, 'augmentations': 171, 'c1_weight': 1.2167576457622766, 'channels': 8, 'epochs': 2, 'layers': 4, 'lr': 0.007072866653232726, 'normal_cell_1_num_ops': 1, 'normal_cell_1_ops_1_input_1': 0, 'normal_cell_1_ops_1_input_2': 0, 'normal_cell_1_ops_1_type': 'MaxPool5', 'normal_cell_1_ops_2_input_1': 1, 'normal_cell_1_ops_2_input_2': 0, 'normal_cell_1_ops_2_type': 'Conv5', 'normal_cell_1_ops_3_input_1': 2, 'normal_cell_1_ops_3_input_2': 0, 'normal_cell_1_ops_3_type': 'Conv7', 'normal_cell_1_ops_4_input_1': 0, 'normal_cell_1_ops_4_input_2': 0, 'normal_cell_1_ops_4_type': 'SepConv3', 'normal_cell_1_ops_5_input_1': 3, 'normal_cell_1_ops_5_input_2': 3, 'normal_cell_1_ops_5_type': 'AvgPool7', 'normal_cell_1_ops_6_input_1': 1, 'normal_cell_1_ops_6_input_2': 0, 'normal_cell_1_ops_6_type': 'MaxPool5', 'normal_cell_1_ops_7_input_1': 0, 'normal_cell_1_ops_7_input_2': 2, 'normal_cell_1_ops_7_type':     'SepConv5', 'normal_cell_1_ops_8_input_2': 6, 'normal_cell_1_ops_8_type': 'Conv3', 'normal_cell_1_ops_9_input_1': 4, 'normal_cell_1_ops_9_input_2': 1, 'normal_cell_1_ops_9_type': 'Conv7', 'num_conv': 1, 'num_re': 1, 'p': 0.02479858526104134, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 'reduction_cell_1_ops_1_type': 'FactorizedReduce'}


    _compute(hyperparameter)

