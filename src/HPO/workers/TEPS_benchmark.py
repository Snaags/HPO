import numpy as np 
import time
import os 
import matplotlib.pyplot as plt
from HPO.utils.model import NetworkMain
from HPO.utils.DARTS_utils import config_space_2_DARTS
from HPO.utils.FCN import FCN 
import pandas as pd
import torch
from HPO.data.teps_datasets import Train_TEPS , Test_TEPS
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import HPO.utils.augmentation as aug
from HPO.utils.train_log import Logger
from HPO.utils.train_utils import collate_fn_padd
from HPO.utils.train import train_model
from HPO.utils.weight_freezing import freeze_FCN, freeze_resnet
from HPO.utils.ResNet1d import resnet18
from HPO.utils.files import save_obj
from queue import Empty
from sklearn.model_selection import StratifiedKFold as KFold
from collections import namedtuple
from HPO.utils.worker_score import Evaluator 
from HPO.utils.worker_utils import LivePlot
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


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
      print("Got Configuration!")

    if device != None:
      print("Starting config with device: {}".format(device))
      complete = False
      crashes = 0
      acc , rec =  _compute(hyperparameter = config , cuda_device = device)
      while not complete:
        try:
          
          complete = True
        except:
          crashes +=1
          print("Model crash: {} ".format(crashes))
          time.sleep(60)
          if crashes == 2:
            print("Final Crash giving score of zero")
            acc , rec = 0 , 0 
            complete = True
      res.put([config , acc , rec ]) 

  torch.cuda.empty_cache()
   
def _compute(hyperparameter,budget = 1, in_model = None , train_path = None,  test_path = None, cuda_device = None,plot_queue = None, model_id = None, binary = True):
  ### Configuration 
  THRESHOLD = 0.4 #Cut off for classification
  batch_size = 16
  if cuda_device == None:
     cuda_device = 1# torch.cuda.current_device()
  
  ##Set up augmentations##
  jitter = aug.Jitter(device = cuda_device,sigma = 0.125, rate = 0.5)
  crop = aug.Crop(device = cuda_device, rate = 0.8, crop_min = 0.1 , crop_max = 0.98)
  scaling = aug.Scaling(device = cuda_device)
  window_warp = aug.WindowWarp(device = cuda_device,rate = 0.9)
  cut_out = aug.CutOut(device = cuda_device)
  mix_up = aug.MixUp(device = cuda_device,m = 0.2, rate = 0.2)
  augmentations = [jitter,crop,scaling, window_warp, cut_out]

  torch.cuda.set_device(cuda_device)

  print("Cuda Device Value: ", cuda_device)
  gen = config_space_2_DARTS(hyperparameter,reduction = True)
  print(gen)
  n_classes = dataset_train.get_n_classes()
  inner = 4
  kfold = KFold(n_splits = inner, shuffle = True)
=======
  logger = Logger()
  budget = 10
  multibatch = False
  for fold in range(budget):
      print('---Fold No.--{}----------------------'.format(fold))
      dataset_train = Train_TEPS(augmentations = augmentations, samples_per_class = 60,device = cuda_device,one_hot = False,binary = True)
      dataset_test = Test_TEPS(binary = True,samples_per_class = 500)
      n_classes = dataset_train.get_n_classes()
      torch.cuda.empty_cache()
      if multibatch:
        batches = [2,4,8,16]
        trainloader = {}
        for i in batches:
          trainloader[i] = torch.utils.data.DataLoader(
                              dataset_train,collate_fn = collate_fn_padd,shuffle = True,
                              batch_size=i, drop_last = True)
      else:
        trainloader = torch.utils.data.DataLoader(
                          dataset_train,collate_fn = collate_fn_padd,shuffle = True,
                          batch_size=batch_size, drop_last = True)
      testloader = torch.utils.data.DataLoader(
                          dataset_test,collate_fn = collate_fn_padd,shuffle = True,
                          batch_size=batch_size,drop_last = True)
      evaluator = Evaluator(batch_size, n_classes,cuda_device,testloader = testloader) 

      model = NetworkMain(dataset_train.get_n_features(),hyperparameter["channels"],num_classes= dataset_train.get_n_classes() , layers = hyperparameter["layers"], auxiliary = False,drop_prob = hyperparameter["p"], genotype = gen, binary = binary)
      model = model.cuda(device = cuda_device)
      """
      ### Train the model
      """
      train_model(model , hyperparameter, trainloader , hyperparameter["epochs"], batch_size , cuda_device, graph = plot_queue, binary = binary,evaluator = evaluator) 
      """
      ### Test the model
      """
      evaluator.forward_pass(model, testloader,binary)
      evaluator.predictions(model_is_binary = binary , THRESHOLD = THRESHOLD)
      evaluator.ROC()
      ### Get Metrics
      total = evaluator.T()
      acc  =  evaluator.T_ACC()
      recall = evaluator.TPR(1)
      recall_total = evaluator.P(1)
      #sup = evaluator.sup_loss(model, testloader)
      #sup10 = evaluator.sup_loss(model, testloader, n_augment = 10)
      #unsup10 = evaluator.unsup_loss(model, testloader, n_augment = 10)
      #unsup = evaluator.unsup_loss(model, testloader)
      #cons = evaluator.c_loss(model, testloader)
      #cons_10 = evaluator.c_loss(model, testloader,10)
      #posttrain_naswot = evaluator.score_naswot(model,testloader)
      #df_loss = evaluator.loss_over_sample_size(model, dataset_test)
      #print("Supervised Loss: {} -- Unsupvised Loss: {}".format(sup,unsup))
      #print("Supervised Loss(10): {} -- Unsupervised Loss(10): {}".format(sup10,unsup10))
      #print("NASWOT PRE: {} -- NASWOT POST: {}".format(pretrain_naswot,posttrain_naswot))
      print("Accuracy: ", "%.4f" % ((acc)*100), "%")
      #print("Recall: ", "%.4f" % ((recall)*100), "%")

      ### Save Model
      def save_model(model, hyperparameter):
        model_zoo = "{}/scripts/model_zoo/".format(os.environ["HOME"])
        torch.save(model.state_dict() , model_zoo+"-Acc-{}-Rec-{}".format(acc, recall))
        save_obj( hyperparameter , model_zoo+"hps/"+"-Acc-{}-Rec-{}".format(acc , recall) )


  print("Final Scores -- ACC: {} -- REC: {}".format(acc, recall))
  return acc, recall, sup.item(), unsup.item()#, sup10.item(),unsup10.item()

if __name__ == "__main__":
  hpo = {'channels': 128, 'lr': 0.0025170869707739693, 'p': 0.00, 'epochs': 200, 'layers': 3}
  hyperparameter = {'normal_index_0_0': 0,
  'normal_index_0_1': 0,
  'normal_index_1_0': 1,
  'normal_index_1_1': 0,
  'normal_index_2_0': 1,
  'normal_index_2_1': 2,
  'normal_index_3_0': 2,
  'normal_index_3_1': 2,
  'normal_node_0_0': 'max_pool_3x3',
  'normal_node_0_1': 'avg_pool_3x3',
  'normal_node_1_0': 'avg_pool_3x3',
  'normal_node_1_1': 'sep_conv_7x7',
  'normal_node_2_0': 'avg_pool_3x3',
  'normal_node_2_1': 'sep_conv_7x7',
  'normal_node_3_0': 'sep_conv_3x3',
  'normal_node_3_1': 'sep_conv_7x7',
  'reduction_index_0_0': 1,
  'reduction_index_0_1': 0,
  'reduction_index_1_0': 0,
  'reduction_index_1_1': 0,
  'reduction_index_2_0': 1,
  'reduction_index_2_1': 2,
  'reduction_index_3_0': 1,
  'reduction_index_3_1': 2,
  'reduction_node_0_0': 'skip_connect',
  'reduction_node_0_1': 'sep_conv_7x7',
  'reduction_node_1_0': 'avg_pool_3x3',
  'reduction_node_1_1': 'max_pool_3x3',
  'reduction_node_2_0': 'sep_conv_7x7',
  'reduction_node_2_1': 'sep_conv_7x7',
  'reduction_node_3_0': 'skip_connect',
  'reduction_node_3_1': 'dil_conv_3x3'}
  hyperparameter.update(hpo)
  _compute(hyperparameter, binary = True)
  exit()

