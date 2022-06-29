import numpy as np 
import time
import os 
import matplotlib.pyplot as plt
from HPO.utils.model_constructor import Model
import pandas as pd
import torch
from HPO.data.datasets import Test_TEPS_split , Train_TEPS_split
import torch.nn as nn
import torch
import random
from HPO.utils.time_series_augmentation import permutation , magnitude_warp, time_warp
from HPO.utils.time_series_augmentation_torch import jitter, scaling, rotation
from HPO.utils import weight_freezing as wf



  
def compute(hyperparameter,budget = 1, in_model = None):
  ###Configuration###
  DATASET_PATH = "/home/snaags/scripts/datasets/TEPS/split"
  #torch.cuda.set_device(1) #Set cuda Device
  ###################
  test_files = []
  train_files = []
  files = os.listdir(DATASET_PATH)
  for i in files:
    if "training" in i:
      train_files.append(i)
    else:
      test_files.append(i)
  train_dataset = Train_TEPS_split(train_files,hyperparameter["window_size"])
  test_dataset = Test_TEPS_split(test_files,hyperparameter["window_size"])
  num_classes =  train_dataset.get_n_classes()
  
  batch_size = 64
  train_dataloader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size,
    shuffle = True,drop_last=True,pin_memory=True)
  test_dataloader = torch.utils.data.DataLoader( test_dataset, batch_size=batch_size,shuffle = True, 
                                 drop_last=True,pin_memory=True)
  if in_model == None:
    model = Model(input_size = ( train_dataset.features,  train_dataset.window),output_size =  num_classes,hyperparameters = hyperparameter)
  else:
    model = in_model  
  model = model.cuda()
  train_dataset.get_n_samples_per_class()
  test_dataset.get_n_samples_per_class()
  
  """
  ## Train the model
  """
  
  ###Training Configuration
  max_iter = 1000
  n_iter =  train_dataset.get_n_samples()/batch_size
  if max_iter < n_iter:
    n_iter = max_iter
  epochs = budget
  print("Memory reported as memory_allocated: ",torch.cuda.memory_reserved(0))
  wf.freeze_normal_cells(model)

  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  criterion = nn.CrossEntropyLoss()
  acc = 0
  peak_acc = 0
  def cal_acc(y,t):
      return np.count_nonzero(y==t)/len(y)
  def convert_label_max_only(y):
      y = y.cpu().detach().numpy()
      idx = np.argmax(y,axis = 1)
      out = np.zeros_like(y)
      for count, i in enumerate(idx):
          out[count, i] = 1
      return idx
  total_label_faults = 0
  total_label_normals = 0
  for epoch in range(epochs):
    for i, (samples, labels) in enumerate( train_dataloader):
      samples = samples.cuda(non_blocking=True)
      labels = labels.cuda(non_blocking=True)
  
      # zero the parameter gradients
      optimizer.zero_grad()
      outputs = model(samples.float())
      # forward + backward + optimize
  
      outputs = outputs
      labels = labels.long()
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      if i%10 == 0:
        acc += cal_acc(convert_label_max_only(outputs), labels.cpu().detach().numpy())
        acc = acc/2
        if acc > peak_acc:
          peak_acc = acc
        # Save the canvas
        print("Epoch (",str(epoch),"/",str(epochs), ")""Iteration(s) (", str(i),"/",str(n_iter), ") Loss: ","%.2f" % loss.item(), "Accuracy: ","%.2f" % acc ,"-- Peak Accuracy: ", peak_acc, end = '\r')
      if i >= max_iter:
        break
  print(" Loss: ","%.2f" % loss.item(), "Accuracy: ","%.2f" % acc )
    
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
        if i % 500 == 0:
          break  
  print() 
  print("Total Test Samples : {} - Total Faults: {}".format(total,recall_total))
  print("Accuracy: ", "%.4f" % ((correct/total)*100), "%")
  print("Recall: ", "%.4f" % ((recall_correct/recall_total)*100), "%")
 


  torch.cuda.empty_cache()
  from HPO.workers.repsol_worker import compute as _compute
  repsol_res =_compute(hyperparameter, 1, model.train())
  print("REPSOL Accuracy: {}".format(repsol_res))

  return (correct/total) ,repsol_res 
  
  

if __name__ == "__main__":
  hyperparameter = {'channels': 27, 'layers': 8, 'lr': 0.0001, 'normal_cell_1_num_ops': 1, 'normal_cell_1_ops_1_input_1': 0, 'normal_cell_1_ops_1_input_2' : 0, 
  'normal_cell_1_ops_1_type': 'Conv3', 'normal_cell_1_ops_2_input_1': 0, 'normal_cell_1_ops_2_input_2': 1, 'normal_cell_1_ops_2_type': 'Identity', 'normal_cell_1_ops_3_input_1': 0, 
  'normal_cell_1_ops_3_input_2': 2, 'normal_cell_1_ops_3_type': 'SepConv7', 'normal_cell_1_ops_4_input_1': 0, 'normal_cell_1_ops_4_input_2': 1, 'normal_cell_1_ops_4_type': 'AvgPool5',
   'normal_cell_1_ops_5_input_1': 2, 'normal_cell_1_ops_5_input_2': 4, 'normal_cell_1_ops_5_type': 'MaxPool7', 'normal_cell_1_ops_6_input_1': 1, 'normal_cell_1_ops_6_input_2': 1, 
   'normal_cell_1_ops_6_type': 'AvgPool7', 'normal_cell_1_ops_7_input_1': 5, 'normal_cell_1_ops_7_input_2': 6, 'normal_cell_1_ops_7_type': 'AvgPool5', 'normal_cell_1_ops_8_input_1': 2, 
   'normal_cell_1_ops_8_input_2': 3, 'normal_cell_1_ops_8_type': 'Identity', 'normal_cell_1_ops_9_input_1': 6, 'normal_cell_1_ops_9_input_2': 1, 'normal_cell_1_ops_9_type': 'MaxPool5',
    'num_conv': 1, 'num_re': 1, 'p': 0.05, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 
    'reduction_cell_1_ops_1_type': 'FactorizedReduce', 'window_size': 499} 
  worker = TEPS_worker()
  worker.compute(hyperparameter)

