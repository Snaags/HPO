import random
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from HPO.utils.model_constructor import Model
from torch.utils.data import DataLoader
from HPO.utils.time_series_augmentation_torch import jitter, scaling , rotation, window_warp, crop, cutout,mix_up, cut_mix
class WeightTest:
  def __init__(self, model):
    self.count_dict = {}
    self.model = model
    for name, param in self.model.named_parameters():
      if 'weight' in name:
        self.count_dict[name] = 0
  def step(self):
    for name, param in self.model.named_parameters():
      if ('weight' in name) and (name != "nets.0.fc.weight"):
        print("Name: {}".format(name))
        temp = torch.zeros(param.grad.shape)
        temp[param.grad != 0] += 1
        self.count_dict[name] += temp
        print("Weight {} updates: {}".format(name, min(self.count_dict[name])))


import gc
def tensor_list():
  for obj in gc.get_objects():
      try:
          if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
              print(type(obj), obj.size())
      except:
          pass

class BarlowTwins(nn.Module):
  def __init__(self, model):
     super().__init__()
     layers = []
     self.nets = nn.ModuleList()
     self.nets.append(model)
     self.model = model
     size = model.get_channels()
     for i in range(2):
         if i ==0:
           layers.append(nn.Linear(size, size*4, bias=False))
         else:
           layers.append(nn.Linear(size*4, size*4, bias=False))
         layers.append(nn.BatchNorm1d(size*4))
         layers.append(nn.ReLU(inplace=True))
     layers.append(nn.Linear(size*4, size*4, bias=False))
     self.projector = nn.Sequential(*layers)
     self.nets.append(self.projector)
  def forward(self , x): 
    x = self.nets[0]._forward(x)
    x = self.nets[1](x)
    return x

class Metaloader:
  def __init__(self, dataloaders,max_epochs):
    pass  

  def __getitem__(self,idx):
    non_empty_dataloaders = list(dataloaders)
    iter_dl = {}
    for b in dataloaders:
      iter_dl[b] = iter(dataloaders[b])
    for i in range(n_iter):
      while True:
        batch_size = random.choice(non_empty_dataloaders)
        try:
          samples, labels = next(iter_dl[batch_size])
        except StopIteration:
          non_empty_dataloaders.remove(batch_size)
        else:
          break


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

def train_model_multibatch(model : Model , hyperparameter : dict, dataloaders : dict , epochs : int, batch_size : int, cuda_device = None, augment_num = 2, graph = None, binary = False):
  """
  Trains models with an assortment of batch sizes to produce a wider range of augmentated samples
  """

  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = hyperparameter["lr_step"])
  #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = hyperparameter["T_0"] , T_mult = hyperparameter["T_mult"])
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
    aug_list = []
    aug_labels = []
    non_empty_dataloaders = list(dataloaders)
    iter_dl = {}
    for b in dataloaders:
      iter_dl[b] = iter(dataloaders[b])
    if epoch < epochs/2:
      non_empty_dataloaders.remove(4)
    if epoch < epochs/2:
      non_empty_dataloaders.remove(8)
    if epoch < 2*(epochs/3):
      non_empty_dataloaders.remove(16)
    if epoch > 3*(epochs/4):
      non_empty_dataloaders.remove(2)
    n_iter = 0
    for b_s in non_empty_dataloaders:
      n_iter += len(dataloaders[b_s])
    for i in range(n_iter):
      while True:
        batch_size = random.choice(non_empty_dataloaders)
        try:
          samples, labels = next(iter_dl[batch_size])
        except StopIteration:
          non_empty_dataloaders.remove(batch_size)
        else:
          break
      samples = samples.cuda(non_blocking=True, device = cuda_device)
      labels = labels.cuda(non_blocking=True, device = cuda_device)
      aug_sample , aug_label = augment(samples,labels,hyperparameter, cuda_device)
      aug_list.append(aug_sample)
      aug_labels.append(aug_label)
      optimizer.zero_grad()
      if batch_size > 1:
        labels = labels.long().view( batch_size  )
      else:
        labels = labels.long().view( 1 )
      outputs = model(samples.float()).cuda(device = cuda_device)
      # forward + backward + optimize
      if binary:
        loss = criterion(outputs.view(batch_size), labels.float()).cuda(device = cuda_device)
      else:
        loss = criterion(outputs, labels).cuda(device = cuda_device)
      loss.backward()
      loss_list.append(loss.item())
      optimizer.step()
      if i %5 == 0:
        if graph != None:
          graph.put(loss_list)
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , correct , total, peak_acc, loss.item(), n_iter, loss_list, binary = binary)


    #tensor_list()
    #scheduler.step()
    epoch += 1 
    #dataloader.set_iterator()
  print()
  print("Num epochs: {}".format(epoch))

def train_model_aug(model : Model , hyperparameter : dict, dataloader : DataLoader , epochs : int, batch_size : int, cuda_device = None, augment_num = 2, graph = None, binary = False):
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = hyperparameter["lr_step"])
  #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = hyperparameter["T_0"] , T_mult = hyperparameter["T_mult"])
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
    aug_list = []
    aug_labels = []
    for i, (samples, labels) in enumerate( dataloader ):
      samples = samples.cuda(non_blocking=True, device = cuda_device)
      labels = labels.cuda(non_blocking=True, device = cuda_device)
      if augment_num:
          aug_sample , aug_label = augment(samples,labels,hyperparameter, cuda_device)
          aug_list.append(aug_sample)
          aug_labels.append(aug_label)
      else:
          aug_list.append(samples)
          aug_labels.append(labels)

    for i , (samples , labels) in enumerate(zip(aug_list , aug_labels)):
      optimizer.zero_grad()
      if batch_size > 1:
        labels = labels.long().view( batch_size  )
      else:
        labels = labels.long().view( 1 )
      outputs = model(samples.float()).cuda(device = cuda_device)
      # forward + backward + optimize
      if binary:
        loss = criterion(outputs.view(batch_size), labels.float()).cuda(device = cuda_device)
      else:
        loss = criterion(outputs, labels).cuda(device = cuda_device)
      loss.backward()
      loss_list.append(loss.item())
      optimizer.step()
      if i %5 == 0:
        if graph != None:
          graph.put(loss_list)
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , correct , total, peak_acc, loss.item(), n_iter, loss_list,binary = binary)


    #tensor_list()
    #scheduler.step()
    epoch += 1 
    #dataloader.set_iterator()
  print()
  print("Num epochs: {}".format(epoch))

def train_model(model : Model , hyperparameter : dict, dataloader : DataLoader , epochs : int, 
    batch_size : int, cuda_device = None, augment_on = 0, graph = None, binary = False,evaluator= None):
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = hyperparameter["lr_step"])
  #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = hyperparameter["T_0"] , T_mult = hyperparameter["T_mult"])
  if binary == True:
    criterion = nn.BCEWithLogitsLoss().cuda(device = cuda_device)
  
  else:
    #if "c1" in hyperparameter:
    #  criterion = nn.CrossEntropyLoss(torch.Tensor([1, hyperparameter["c1"]])).cuda(device = cuda_device)
    #else:
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
      #samples = samples.cuda(non_blocking=True, device = cuda_device)
      #labels = labels.cuda(non_blocking=True, device = cuda_device)
      batch_size = samples.shape[0]
      # zero the parameter gradients
      optimizer.zero_grad()
      #if batch_size > 1:
      #  labels = labels.long().view( batch_size  )
      #else:
      #  labels = labels.long().view( 1 )
      outputs = model(samples.float()).cuda(device = cuda_device)
      # forward + backward + optimize
      if batch_size == 1:
        outputs = outputs.view(batch_size,outputs.shape[0])
      if binary:
        loss = criterion(outputs.view(batch_size), labels.float()).cuda(device = cuda_device)
      else:
        loss = criterion(outputs, labels).cuda(device = cuda_device)
      loss.backward()
      loss_list.append(loss.item())
      optimizer.step()
      if i %10 == 0:
        if graph != None:
          graph.put(loss_list)

      if i % 500 == 0 and i != 0:
        if evaluator != None:
          evaluator.forward_pass(model,subset = 200)
          evaluator.predictions(model_is_binary = False)
          acc  =  evaluator.T_ACC()
          evaluator.reset_cm()
          print("Validation set Accuracy: {}".format(acc))
      if i% 5 == 0:
        #correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , correct , total, peak_acc, loss.item(), n_iter, loss_list)
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , correct , total, peak_acc, loss.item(), n_iter, loss_list,binary = binary)


    #scheduler.step()
    epoch += 1
    #dataloader.set_iterator()
  print()
  print("Num epochs: {}".format(epoch))



def train_model_bt(model : Model , hyperparameter : dict, dataloader : DataLoader , epochs : int, batch_size : int, cuda_device = None, validation_set = None, graph = None):
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 
  bt = BarlowTwins(model).cuda(device = cuda_device)
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  criterion = barlow_twins
  epoch = 0
  loss_list = []
  #weight_test = WeightTest(bt)
  peak_acc = 0
  while epoch < epochs:
    for i, samples in enumerate( dataloader ):
      # zero the parameter gradients
      optimizer.zero_grad()
      loss = criterion(bt , samples, cuda_device = cuda_device)
      # forward + backward + optimize
      loss.backward()
      loss_list.append(loss.item())
      #weight_test.step()
      optimizer.step()
      if i %3 == 0:
        if graph != None:
          graph.put(loss_list)
        print("Epoch: {}/{} - Iteration: {}/{} - Loss: {} ".format(epoch , epochs,i, n_iter , loss.item()))
    epoch += 1 
    #dataloader.set_iterator()
  print()
  print("Num epochs: {}".format(epoch))





def augment(x, y,hp, device = None):
  augs = [jitter, scaling, window_warp,crop, mix_up,cutout]

  if "jitter" in hp:
    args = [hp["jitter"], hp["scaling"], hp["window_warp_num"], hp["crop"] , hp["mix_up"], hp["cut_out"]]
    rates = [hp["jitter_rate"], hp["scaling_rate"], hp["window_warp_rate"],hp["crop_rate"], hp["mix_up_rate"], hp["cut_out_rate"]]

    if device == None:
      for func,arg,rate in zip(augs,args, rates):

        if rate > 1:
          for _ in range((int(rate)+1)):
            if random.random() < (rate/(int(rate)+1)):
              x,y = func(x, y, arg )
        else:
          if random.random() < rate:
            x,y = func(x,y, arg )

    else:
      for func,arg,rate in zip(augs,args, rates):
        if rate > 1:
          for _ in range((int(rate)+1)):
            if random.random() < (rate/(int(rate)+1)):
              x,y = func(x, y, arg )
        else:
          if random.random() < rate:
            x,y = func(x,y, arg , device = device)

    return x,y

  else:
    rate = 0.5
    if device == None:
      for func in augs:
        if random.random() < rate:
          x,y = func(x,y)
    else:
      for func in augs:
        if random.random() < rate:
          x,y = func(x,y, device = device)
    return x ,y 
def barlow_twins(model, batch, cuda_device = None):
  N = batch.shape[0] #Batch Size
  lmbda = 0.005
  #Generate Augmented
  y_a = augment(batch)  
  y_b = augment(batch)
  
  #Generate network embeddings
  z_a = model(y_a.cuda(device = cuda_device).float())
  z_b = model(y_b.cuda(device = cuda_device).float())
  D = z_a.shape[1] #Output Dim

  #Normalise
  z_a_norm = (z_a - z_a.mean(axis = 0)) / z_a.std(axis = 0)
  z_b_norm = (z_b - z_b.mean(axis = 0)) / z_b.std(axis = 0)

  c = torch.matmul(z_a_norm.T , z_b_norm) / N
  idn = torch.eye(D).cuda(device = cuda_device)
  c_diff = torch.pow((c - idn), 2)
  
  #Seperate diag
  c_diff_diag = torch.matmul(c_diff, idn)
  c_diff_non_diag = torch.matmul( c_diff , torch.ones(D,D).cuda(device = cuda_device) - idn )
  c_diff_non_diag = c_diff_non_diag * lmbda  
  c_diff = c_diff_diag + c_diff_non_diag

  loss = c_diff.sum()
  
  return loss






