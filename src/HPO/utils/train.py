import random
import torch
import torch.nn as nn
import numpy as np
from HPO.utils.triplet import Batch_All_Triplet_Loss as Triplet_Loss
from torch import Tensor
from HPO.utils.train_log import Logger
from HPO.utils.model_constructor import Model
from torch.utils.data import DataLoader
from HPO.utils.train_utils import stdio_print_training_data

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
  elif len(labels.shape) > 1:
    new_correct , new_total =  cal_acc(convert_label_max_only(outputs), convert_label_max_only(labels))
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

def train_model_triplet(model : Model , hyperparameter : dict, dataloader : DataLoader , epochs : int, 
    batch_size : int, cuda_device = None, augment_on = 0, graph = None, binary = False,evaluator= None,logger = None,run = None):
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max",patience = 4,verbose = True, factor = 0.1,cooldown = 2,min_lr = 0.0000000000000001)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
  criterion = Triplet_Loss(m = 0.1 , device = cuda_device)
  epoch = 0
  peak_acc = 0
  loss_list = []
  total = 0
  correct = 0
  acc = 0
  recall = 0
  if logger == None:
    logger = Logger()
  while epoch < epochs:
    loss_values = 0
    if epoch % 3 == 0:
      total = 0
      correct = 0
    for i, (samples, labels) in enumerate( dataloader ):
      batch_size = samples.shape[0]
      optimizer.zero_grad()
      outputs = model(samples)
      if binary:
        loss = criterion(outputs.view(batch_size), labels.float()).cuda(device = cuda_device)
      else:
        loss = criterion(outputs, labels).cuda(device = cuda_device)
      loss.backward()
      optimizer.step()
      loss_values += criterion.get_fraction_pos()
      if i % 5 == 0:
        print("Epoch {} - [{}/{}] loss over epoch: {}".format(epoch,i,len(dataloader),loss_values/i))
    if epoch % 1 == 0 and run != None:
        torch.save(model.state_dict() ,"SWA/run-{}-checkpoint-{}".format(run, epoch))
    if epoch % 5 == 0:
        torch.save(model.state_dict() ,"triplet-{}".format(epoch))
        if evaluator != None:
          model.eval()
          evaluator.forward_pass(model,binary = binary)
          evaluator.predictions(model_is_binary = binary,THRESHOLD = 0.4)
          if binary:
            evaluator.ROC("train")
          acc = evaluator.T_ACC()
          recall = evaluator.TPR(1)
          evaluator.reset_cm()
          model.train()
          print("")
          print("Validation set Accuracy: {} -- Recall: {}".format(acc,recall))
          print("")
    scheduler.step()
    epoch += 1
  print()
  print("Num epochs: {}".format(epoch))
  return logger

def train_model(model : Model , hyperparameter : dict, dataloader : DataLoader ,
     cuda_device = None, evaluator= None,logger = None,run = None):

  #INITIALISATION
  EPOCHS = hyperparameter["EPOCHS"]
  BATCH_SIZE = hyperparameter["BATCH_SIZE"] 
  BINARY = hyperparameter["BINARY"]
  PRINT_RATE_TRAIN = hyperparameter["PRINT_RATE_TRAIN"]
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 

  #CONFIGURATION OF OPTIMISER AND LOSS FUNCTION
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["LR"])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS)
  if hyperparameter["BINARY"] == True:
    criterion = nn.BCEWithLogitsLoss().cuda(device = cuda_device)
  else:
    criterion = nn.CrossEntropyLoss().cuda(device = cuda_device)

  #INITIALISE TRAINING VARIABLES
  epoch = 0
  peak_acc = 0
  loss_list = []
  total = 0
  correct = 0
  acc = 0
  recall = 0
  if logger != False or logger == None:
    logger = Logger()

  #MAIN TRAINING LOOP
  while epoch < EPOCHS:
    if epoch % 3 == 0:
      total = 0
      correct = 0
    for i, (samples, labels) in enumerate( dataloader ):
      optimizer.zero_grad()
      outputs = model(samples)
      if BINARY == True:
        loss = criterion(outputs.view(BATCH_SIZE), labels.float())
      else:
        loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      if i % PRINT_RATE_TRAIN == 0 and PRINT_RATE_TRAIN:
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,EPOCHS , correct , total, peak_acc, loss.item(), n_iter, loss_list,binary = BINARY)
      if logger != False:
        logger.update({"loss": loss.item(), "training_accuracy": (correct/total),"index" : i,
              "epoch": epoch, "validation_accuracy": acc, "lr":optimizer.param_groups[0]['lr'],"validation recall": recall })
    if hyperparameter["WEIGHT_AVERAGING_RATE"] and epoch % hyperparameter["WEIGHT_AVERAGING_RATE"] == 0:
        torch.save(model.state_dict() ,"SWA/run-{}-checkpoint-{}".format(run, epoch))
    if epoch % hyperparameter["MODEL_VALIDATION_RATE"] == 0 and hyperparameter["MODEL_VALIDATION_RATE"] and epoch != 0:
        if evaluator != None:
          model.eval()
          evaluator.forward_pass(model,binary = BINARY)
          evaluator.predictions(model_is_binary = BINARY,THRESHOLD = hyperparameter["THRESHOLD"])
          #if binary:
          #  evaluator.ROC("train")
          acc = evaluator.T_ACC()
          recall = evaluator.TPR(1)
          evaluator.reset_cm()
          model.train()
          print("")
          print("Validation set Accuracy: {} -- Recall: {}".format(acc,recall))
          print("")
    #scheduler.step()
    epoch += 1
  print()
  print("Num epochs: {}".format(epoch))
  return logger

def auto_train_model(model : Model , hyperparameter : dict, dataloader : DataLoader ,validation_dataloader ,epochs : int, 
    batch_size : int, cuda_device = None, augment_on = 0, graph = None, binary = False,evaluator= None,logger = None):
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  if binary == True:
    criterion = nn.BCEWithLogitsLoss().cuda(device = cuda_device)
  else:
    criterion = nn.CrossEntropyLoss()
  autotrainer = AutoTrainer(10)
  epoch = 0
  peak_acc = 0
  loss_list = []
  val_loss_list = []
  total = 0
  correct = 0
  acc = 0
  recall = 0
  roc = 0
  if logger == None:
    logger = Logger()
  while epoch < epochs:
    if epoch % 3 == 0:
      total = 0
      correct = 0
    for i, (samples, labels) in enumerate( dataloader ):
      sample_val , label_val = next(iter(validation_dataloader))
      batch_size = samples.shape[0]
      optimizer.zero_grad()
      outputs = model(samples.float()).cuda(device = cuda_device)
      outputs_val = model(sample_val.float()).cuda(device = cuda_device)
      if binary:
        loss = criterion(outputs.view(batch_size), labels.float()).cuda(device = cuda_device)
        loss_val = criterion(outputs_val.view(batch_size), labels_val.float()).cuda(device = cuda_device)
      else:
        loss = criterion(outputs, labels).cuda(device = cuda_device)
        loss_val = criterion(outputs_val, labels_val).cuda(device = cuda_device)
      loss_list.append(loss)
      val_loss_list.append(loss_val) 
      loss.backward()
      optimizer.step()
      if i % 3 ==0 and i != 0:
        f1 = torch.Tensor(loss_list)
        f2 = torch.Tensor(val_loss_list)
        f3 = torch.stack(f1,f2)
        lr = autotrainer(f3)
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
        autotrainer.update(f2[-1])

      if i% 5 == 0:
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , correct , total, peak_acc, loss.item(), n_iter, loss_list,binary = binary)
    
      if i% 5 == 0:
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , correct , total, peak_acc, loss.item(), n_iter, loss_list,binary = binary)
      logger.update({"loss": loss.item(), "training_accuracy": (correct/total),"index" : i,
              "epoch": epoch, "validation_accuracy": acc, "lr":optimizer.param_groups[0]['lr'],"validation recall": recall , "ROC": roc})
    if epoch % 5 == 0:
      if evaluator != None:
        evaluator.forward_pass(model,subset = 450,binary = binary)
        evaluator.predictions(model_is_binary = binary,THRESHOLD = 0.4)
        if binary:
          evaluator.ROC("train")
        acc = evaluator.T_ACC()
        roc =  evaluator.ROC("train")
        acc  =  evaluator.T_ACC()
        recall = evaluator.TPR(1)
        evaluator.reset_cm()
        print("")
        print("Validation set Accuracy: {} -- Recall: {}".format(acc,recall))
        print("")
        scheduler.step(acc)
    epoch += 1
  print()
  print("Num epochs: {}".format(epoch))
  return logger


