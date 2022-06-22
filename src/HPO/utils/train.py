import random
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from HPO.utils.model_constructor import Model
from torch.utils.data import DataLoader

def stdio_print_training_data( iteration : int , outputs : Tensor, labels : Tensor , epoch : int, epochs : int, correct :int , total : int , peak_acc : float , loss : Tensor, n_iter, loss_list = None, binary = True):
  def cal_acc(y,t):
    c = np.count_nonzero(y.T==t)
    tot = len(t)
    return c , tot

  def convert_label_max_only(y):
    y = y.cpu().detach().numpy()
    idx = np.argmax(y,axis = 1)
    out = np.zeros_like(y)
    for count, i in enumerate(idx):
        out[count, i] = 1
    return idx
  if binary == True:
    new_correct , new_total = cal_acc(outputs.ge(0.5).cpu().detach().numpy(), labels.cpu().detach().numpy())
  else:
    new_correct , new_total =  cal_acc(convert_label_max_only(outputs), convert_label_max_only(labels))
  correct += new_correct 
  total += new_total
  acc = correct / total 
  if acc > peak_acc:
    peak_acc = acc
  # Save the canvas
  print("Epoch (",str(epoch),"/",str(epochs), ") Accuracy: ","%.2f" % acc, "Iteration(s) (", str(iteration),"/",str(n_iter), ") Loss: "
    ,"%.2f" % loss," Correct / Total : {} / {} ".format(correct , total),  end = '\r')
  return correct ,total ,peak_acc


def train_model(model : Model , hyperparameter : dict, dataloader : DataLoader , epochs : int, 
    batch_size : int, cuda_device = None, augment_on = 0, graph = None, binary = False,evaluator= None):
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",patience = 5,verbose = True, factor = 0.5,cooldown = 2)
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
        if evaluator != None:
          evaluator.forward_pass(model,subset = 50)
          evaluator.predictions(model_is_binary = False)
          acc  =  evaluator.T_ACC()
          evaluator.reset_cm()
          print("")
          print("Validation set Accuracy: {}".format(acc))
          print("")
          scheduler.step(acc)
      if i% 5 == 0:
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , correct , total, peak_acc, loss.item(), n_iter, loss_list,binary = binary)
    epoch += 1
  print()
  print("Num epochs: {}".format(epoch))


