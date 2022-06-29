import random
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from HPO.utils.model_constructor import Model
from torch.utils.data import DataLoader
from HPO.utils.train_utils import stdio_print_training_data
def train_model(model : Model , hyperparameter : dict, dataloader : DataLoader , epochs : int, 
    batch_size : int, cuda_device = None, augment_on = 0, graph = None, binary = False,evaluator= None):
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max",patience = 6,verbose = True, factor = 0.5,cooldown = 2)
  if binary == True:
    criterion = nn.BCEWithLogitsLoss().cuda(device = cuda_device)
  
  else:
    criterion = nn.CrossEntropyLoss()
  epoch = 0
  peak_acc = 0
  loss_list = []
  total = 0
  correct = 0
  while epoch < epochs:
    if epoch % 3 == 0:
      total = 0
      correct = 0
    for i, (samples, labels) in enumerate( dataloader ):
      batch_size = samples.shape[0]
      optimizer.zero_grad()
      outputs = model(samples.float()).cuda(device = cuda_device)
      if binary:
        loss = criterion(outputs.view(batch_size), labels.float()).cuda(device = cuda_device)
      else:
        loss = criterion(outputs, labels).cuda(device = cuda_device)
      loss.backward()
      loss_list.append(loss.item())
      optimizer.step()
      if i %100 == 0:
        if graph != None:
          graph.put(loss_list)

      if i % 100 == 0 and i != 0:
      if i% 5 == 0:
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , correct , total, peak_acc, loss.item(), n_iter, loss_list,binary = binary)
    
      if evaluator != None:
        evaluator.forward_pass(model,subset = 50)
        evaluator.predictions(model_is_binary = False)
        val_acc  =  evaluator.T_ACC()
        evaluator.reset_cm()
        print("")
        print("Validation set Accuracy: {}".format(val_acc))
        print("")
        scheduler.step(val_acc)
    train_acc = correct/total
    [loss_list,val_acc,train_acc,]
    epoch += 1
  print()
  print("Num epochs: {}".format(epoch))


