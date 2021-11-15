
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from HPO.utils.model_constructor import Model
from torch.utils.data import DataLoader

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
     
    labels = torch.Tensor([t[1] for t in batch])
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

def stdio_print_training_data( iteration : int , outputs : Tensor, labels : Tensor , epoch : int, epochs : int, correct :int , total : int , peak_acc : float , loss : Tensor, n_iter):
  def cal_acc(y,t):
    correct = np.count_nonzero(y==t)
    total = len(t)
    return correct , total

  def convert_label_max_only(y):
    y = y.cpu().detach().numpy()
    idx = np.argmax(y,axis = 1)
    out = np.zeros_like(y)
    for count, i in enumerate(idx):
        out[count, i] = 1
    return idx

  new_correct , new_total =  cal_acc(convert_label_max_only(outputs), labels.cpu().detach().numpy())
  correct += new_correct 
  total += new_total
  acc = correct / total 
  if acc > peak_acc:
    peak_acc = acc
  # Save the canvas
  print("Epoch (",str(epoch),"/",str(epochs), ") Accuracy: ","%.2f" % acc, "Iteration(s) (", str(iteration),"/",str(n_iter), ") Loss: "
    ,"%.2f" % loss," Correct / Total : {} / {} ".format(correct , total),  end = '\r')
  return correct ,total ,peak_acc



def train_model(model : Model , hyperparameter : dict, dataloader : DataLoader , epochs : int, batch_size : int, cuda_device = None):
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = hyperparameter["lr_step"])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = hyperparameter["T_0"] , T_mult = hyperparameter["T_mult"])
  if "c1" in hyperparameter:
    criterion = nn.CrossEntropyLoss(torch.Tensor([1, hyperparameter["c1"]]))
  criterion = nn.CrossEntropyLoss()
  epoch = 0
  peak_acc = 0

  total = 0
  correct = 0
  while epoch < epochs:
    if epoch % 3 == 0:
      total = 0
      correct = 0
    for i, (samples, labels) in enumerate( dataloader ):
      # zero the parameter gradients
      optimizer.zero_grad()
      samples = samples.cuda(non_blocking=True, device = cuda_device)
      labels = labels.cuda(non_blocking=True, device = cuda_device)
      if batch_size > 1:
        labels = labels.long().view( batch_size  )
      else:
        labels = labels.long().view( 1 )
      outputs = model(samples.float()).cuda(device = cuda_device)
      # forward + backward + optimize
      if batch_size == 1:
        outputs = outputs.view(batch_size,outputs.shape[0])
      loss = criterion(outputs, labels).cuda(device = cuda_device)
      loss.backward()
      optimizer.step()
      if i %5 == 0:
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , correct , total, peak_acc, loss.item(), n_iter)

    scheduler.step()
    epoch += 1 
    #dataloader.set_iterator()
  print()
  print("Num epochs: {}".format(epoch))
