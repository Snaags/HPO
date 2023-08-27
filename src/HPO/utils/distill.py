import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import json
from HPO.utils.triplet import Batch_All_Triplet_Loss as Triplet_Loss
from torch import Tensor
from HPO.utils.train_log import Logger
from HPO.utils.utils import calculate_train_vals
from HPO.utils.model_constructor import Model
from torch.utils.data import DataLoader
from HPO.utils.train import stdio_print_training_data
from sklearn.metrics import confusion_matrix

def clone_state_dict(state_dict):
    return {name: val.clone() for name, val in state_dict.items()}

def average_state_dicts(state_dicts):
    avg_state_dict = {name: torch.stack([d[name] for d in state_dicts], dim=0).mean(dim=0)
                      for name in state_dicts[0] if state_dicts[0][name].dtype.is_floating_point}
    return avg_state_dict

def train_distill(model : Model , hyperparameter : dict, dataloader : DataLoader ,teacher,
     cuda_device = None, evaluator= None,logger = None,run = None,fold = None, repeat = None):

  params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  #INITIALISATION
  EPOCHS = hyperparameter["EPOCHS"]
  BATCH_SIZE = hyperparameter["BATCH_SIZE"] 
  BINARY = hyperparameter["BINARY"]
  PRINT_RATE_TRAIN = hyperparameter["PRINT_RATE_TRAIN"]
  if cuda_device == None:
    cuda_device = torch.cuda.current_device()
  n_iter = len(dataloader) 

  #CONFIGURATION OF OPTIMISER AND LOSS FUNCTION
  optimizer = torch.optim.AdamW(model.parameters(),lr = hyperparameter["LR"],weight_decay = 0.01)
  if hyperparameter["SCHEDULE"] == True:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min = hyperparameter["LR_MIN"])
  if hyperparameter["BINARY"] == True:
    criterion = nn.BCEWithLogitsLoss().cuda(device = cuda_device)
  else:
    criterion = nn.CrossEntropyLoss().cuda(device = cuda_device)
    dist_loss = nn.KLDivLoss(reduction="batchmean").cuda(device = cuda_device)
    aux_criterion = nn.MSELoss().cuda(device = cuda_device)

  #INITIALISE TRAINING VARIABLES
  epoch = 0
  alpha = 0.9
  beta = 0.1
  peak_acc = 0
  loss_list = []
  total = 0
  correct = 0
  cm_test = ""
  acc = 0
  val_loss = torch.Tensor([0])
  recall = 0
  val_acc = 0
  if hyperparameter["LOGGING"]:
    logger = Logger(hyperparameter["database"],hyperparameter["experiment"],hyperparameter["DATASET_CONFIG"]["NAME"],fold,repeat,params)
  else:
    logger = None

  #model.disable_dropout()
  #MAIN TRAINING LOOP
  while epoch < EPOCHS:
    if epoch % 3 == 0:
      total = 0
      pred_tensor = torch.Tensor()
      gt_tensor = torch.Tensor()
      correct = 0
    weights =[]
    for i, (samples, labels,auxy) in enumerate( dataloader ):
      optimizer.zero_grad()
      #samples, labels = samples.cuda(cuda_device).float(), labels.cuda(cuda_device).long()
      outputs,aux_output = model(samples)
      t_pred, t_aux = teacher.teacher(samples,hyperparameter["T"])
      loss1 = hyperparameter["T"]**2 * dist_loss(F.log_softmax(outputs, dim=1),t_pred )
      loss2 = criterion(outputs,  labels.long())
      loss3 = aux_criterion(aux_output,  t_aux)
      loss4 = aux_criterion(aux_output,  auxy)
      loss = (alpha*loss1+beta*loss2)*0.5 + (alpha*loss3+beta*loss4)*0.5
      loss.backward()
      optimizer.step()
      if hyperparameter["WEIGHT_AVERAGING_RATE"] and epoch > EPOCHS/2:
        weights.append(clone_state_dict(model.state_dict()))
      if hyperparameter["LOGGING"]:
        pred_tensor = torch.cat((pred_tensor, outputs.detach().cpu().flatten(end_dim = 0)))
        gt_tensor = torch.cat((gt_tensor, labels.detach().cpu().flatten()))

      if PRINT_RATE_TRAIN and i % PRINT_RATE_TRAIN == 0:
        correct , total, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,EPOCHS , correct , total, peak_acc, loss.item(), n_iter, loss_list,binary = BINARY)

    if hyperparameter["LOGGING"]:
      cm_train, train_acc = calculate_train_vals(pred_tensor,gt_tensor)
      if PRINT_RATE_TRAIN:
        pred_labels = torch.argmax(pred_tensor, dim=1).numpy()
        gt_labels = gt_tensor.numpy()
        with np.printoptions(linewidth = (10*len(np.unique(gt_labels))+20),precision=4, suppress=True):
          print(confusion_matrix(gt_labels,pred_labels))

    if hyperparameter["MODEL_VALIDATION_RATE"] and epoch % hyperparameter["MODEL_VALIDATION_RATE"] == 0:
        if evaluator != None:
          model.eval()
          evaluator.forward_pass(model,binary = BINARY)
          evaluator.predictions(model_is_binary = BINARY,THRESHOLD = hyperparameter["THRESHOLD"], no_print = False)
          val_acc = evaluator.T_ACC()
          recall = evaluator.TPR(1)
          val_loss = evaluator.calculate_loss(aux_criterion,BINARY)
          cm_test = evaluator.confusion_matrix.copy()
          bal_acc = evaluator.balanced_acc()
          evaluator.reset_cm()
          model.train()
          print("")
          print("Validation set Accuracy: {} -- Balanced Accuracy: {} -- loss: {}".format(val_acc, bal_acc ,val_loss))
          print("")

    if hyperparameter["LOGGING"]:
      cm_test = json.dumps(cm_test.tolist())
      logger.update({
        "loss": loss.item(), 
        "training_accuracy": train_acc,
        "ID":hyperparameter["ID"],
        "epoch": epoch, 
        "validation_accuracy": val_acc, 
        "lr":optimizer.param_groups[0]['lr'],
        "validation_loss": val_loss.item(),
        "confusion_matrix_train": cm_train,
        "confusion_matrix_test": cm_test
        })


    if hyperparameter["SCHEDULE"] == True:
      scheduler.step()
    epoch += 1
    if hyperparameter["WEIGHT_AVERAGING_RATE"] and epoch > EPOCHS/2:  
      model.load_state_dict(average_state_dicts(weights))
  #print()
  #print("Num epochs: {}".format(epoch))
  if hyperparameter["LOGGING"]:
    logger.close()
  return logger
