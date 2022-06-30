import random
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from HPO.utils.model_constructor import Model
from torch.utils.data import DataLoader

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

def stdio_print_training_data( iteration : int , outputs : Tensor, labels : Tensor , 
            epoch : int, epochs : int, correct :int , total : int , peak_acc : float , 
            loss : Tensor, n_iter, loss_list = None, binary = True):

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
  print("Epoch (",str(epoch),"/",str(epochs), ") Accuracy: ","%.2f" % acc, 
              "Iteration(s) (", str(iteration),"/",str(n_iter), ") Loss: ",
              "%.2f" % loss," Correct / Total : {} / {} ".format(correct , total),  end = '\r')

  return correct ,total ,peak_acc


def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t[0].shape[1] for t in batch ])    ## padd
    batch_samples = [ torch.transpose(t[0],0,1) for t in batch ]
    batch_samples = torch.nn.utils.rnn.pad_sequence(batch_samples ,batch_first = True)
    ## compute mask
    mask = (batch != 0)
    labels = []
    for pair in batch:
      one_hot_label = pair[1]
      labels.append(one_hot_label)
    labels = torch.stack(labels).squeeze()
    #labels = torch.Tensor([t[1] for t in batch])
    batch = torch.transpose(batch_samples , 1 , 2 )
    return batch, labels

def collate_fn_padd_x(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[1] for t in batch ])    ## padd
    batch_samples = [ torch.transpose(t,0,1) for t in batch ]
    batch_samples = torch.nn.utils.rnn.pad_sequence(batch_samples ,batch_first = True)
    ## compute mask
    mask = (batch != 0)
     

    batch = torch.transpose(batch_samples , 1 , 2 )
    return batch
