import numpy as np 
import time
import os 
import matplotlib.pyplot as plt
from utils.model_constructor import Model
import pandas as pd
import torch
from data.datasets import Test_repsol , Train_repsol
import torch.nn as nn
import torch
import random
from utils.time_series_augmentation import permutation , magnitude_warp, time_warp
from utils.time_series_augmentation_torch import jitter, scaling, rotation
"""
## Build a model

We build a Fully Convolutional Neural Network originally proposed in
[this paper](https://arxiv.org/abs/1611.06455).
The implementation is based on the TF 2 version provided
[here](https://github.com/hfawaz/dl-4-tsc/).
The following hyperparameters (kernel_size, filters, the usage of BatchNorm) were found
via random search using [KerasTuner](https://github.com/keras-team/keras-tuner).

"""
  
def compute(hyperparameter,budget = 1):
  TRAIN_SPLIT = 0.7
  train_files = []
  files = os.listdir("/home/snaags/scripts/datasets/repsol_np")
  for i in range(int(len(files)*TRAIN_SPLIT)):
    train_files.append(files.pop(random.randint(0,len(files)-1)))
  train_dataset = Train_repsol(train_files,hyperparameter["window_size"])
  
  test_dataset = Test_repsol(files,hyperparameter["window_size"])
  num_classes =  train_dataset.get_n_classes()
  
  print(hyperparameter)
  batch_size = 64
  #augmentation_weights = [hyperparameter["jitter_weight"], hyperparameter["scaling_weight"], hyperparameter["rotation_weight"],
  #        hyperparameter["permutation_weight"], hyperparameter["magnitude_weight"], hyperparameter["time_weight"],hyperparameter["window_weight"]]
  # train_dataset.set_window_size(hyperparameter["window_size"])
  # test_dataset.set_window_size(hyperparameter["window_size"])
  train_dataloader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size,
    shuffle = True,drop_last=True,pin_memory=True)
  test_dataloader = torch.utils.data.DataLoader( test_dataset, batch_size=batch_size, 
                                 drop_last=True,pin_memory=True)
  model = Model(input_size = ( train_dataset.features,  train_dataset.window),output_size =  num_classes,hyperparameters = hyperparameter)
  model = model.cuda()
  
  """
  ## Train the model
  """
  
  ###Training Configuration
  max_iter = 10000
  n_iter =  train_dataset.get_n_samples()/batch_size
  if max_iter < n_iter:
    n_iter = max_iter
  epochs = budget
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  criterion = nn.CrossEntropyLoss(torch.Tensor([1.0, hyperparameter["c1_weight"]]).cuda())
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
      #samples = augmentation(samples,  augmentations, augmentation_weights)
      samples = samples.cuda(non_blocking=True)
      labels = labels.cuda(non_blocking=True)
  
      # zero the parameter gradients
      optimizer.zero_grad()
      outputs = model(samples.float())
      #total_label_faults += (labels == 1 ).sum()
      #total_label_normals += (labels == 0 ).sum()
  
      # forward + backward + optimize
  
      outputs = outputs
      labels = labels.long()
      #labels = labels.unsqueeze(-1)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      if i%50 == 0:
        ##hl logging
        #print("Faults: ", total_label_faults)
        #print("Non-Faults: ", total_label_normals)
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
  
        if i % 100 == 0:
          print("Recall: ", recall_correct/recall_total)
          print("Acc: ",correct/total) 
          print("total faults: ", recall_total) 
          print("total faults correct: ", recall_correct) 
  print() 
  print("Accuracy: ", "%.4f" % ((correct/total)*100), "%")
  print("Recall: ", "%.4f" % ((recall_correct/recall_total)*100), "%")
  
  def plot(output_values):
    c_list = []
    for i in range(output_values.shape[0]):
      if output_values[i]: 
        c_list.append("g")
      else:
        c_list.append( "r")
      if i %100==0:
        print(i , output_values.shape[0])
    import matplotlib.pyplot as plt 
    plt.plot( test_dataset.y, c = c_list)
  
    plt.show()
  return (correct/total) 
  
  

if __name__ == "__main__":
  hyperparameter = {'channels': 27, 'layers': 4, 'lr': 0.0001, 'normal_cell_1_num_ops': 1, 'normal_cell_1_ops_1_input_1': 0, 'normal_cell_1_ops_1_input_2': 0, 'normal_cell_1_ops_1_type': 'Conv3', 'normal_cell_1_ops_2_input_1': 0, 'normal_cell_1_ops_2_input_2': 1, 'normal_cell_1_ops_2_type': 'Identity', 'normal_cell_1_ops_3_input_1': 0, 'normal_cell_1_ops_3_input_2': 2, 'normal_cell_1_ops_3_type': 'SepConv7', 'normal_cell_1_ops_4_input_1': 0, 'normal_cell_1_ops_4_input_2': 1, 'normal_cell_1_ops_4_type': 'AvgPool5', 'normal_cell_1_ops_5_input_1': 2, 'normal_cell_1_ops_5_input_2': 4, 'normal_cell_1_ops_5_type': 'MaxPool7', 'normal_cell_1_ops_6_input_1': 1, 'normal_cell_1_ops_6_input_2': 1, 'normal_cell_1_ops_6_type': 'AvgPool7', 'normal_cell_1_ops_7_input_1': 5, 'normal_cell_1_ops_7_input_2': 6, 'normal_cell_1_ops_7_type': 'AvgPool5', 'normal_cell_1_ops_8_input_1': 2, 'normal_cell_1_ops_8_input_2': 3, 'normal_cell_1_ops_8_type': 'Identity', 'normal_cell_1_ops_9_input_1': 6, 'normal_cell_1_ops_9_input_2': 1, 'normal_cell_1_ops_9_type': 'MaxPool5', 'num_conv': 1, 'num_re': 1, 'p': 0.05, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 'reduction_cell_1_ops_1_type': 'FactorizedReduce', 'window_size': 354} 
  worker = repsol_worker()
  worker.compute(hyperparameter)

