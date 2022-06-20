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
from HPO.utils.time_series_augmentation import permutation , magnitude_warp, time_warp
from HPO.utils.time_series_augmentation_torch import jitter, scaling, rotation
from HPO.utils.worker_train import train_model, collate_fn_padd, train_model_bt, collate_fn_padd_x, train_model_aug, train_model_multibatch
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
  batch_size = 32
  if cuda_device == None:
     cuda_device = 0# torch.cuda.current_device()

  dataset_train = Train_TEPS(samples_per_class = 50)
  dataset_test = Test_TEPS(samples_per_class = 2)
  dataset_test_full = Test_TEPS()
  torch.cuda.set_device(cuda_device)

  print("Cuda Device Value: ", cuda_device)
   
  gen = config_space_2_DARTS(hyperparameter,reduction = True)
  n_classes = dataset_train.get_n_classes()
  inner = 4
  kfold = KFold(n_splits = inner, shuffle = True)
  evaluator = Evaluator(batch_size, n_classes,cuda_device) 
  multibatch = False
  for fold in range(budget):
      print('---Fold No.--{}----------------------'.format(fold))
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
                          dataset_test,collate_fn = collate_fn_padd,
                          batch_size=batch_size,drop_last = True)

      testloader_full = torch.utils.data.DataLoader(
                          dataset_test_full,collate_fn = collate_fn_padd,
                          batch_size=batch_size,drop_last = True)
      ### Build Model
      model = NetworkMain(dataset_train.get_n_features(),hyperparameter["channels"],num_classes= dataset_train.get_n_classes() , layers = hyperparameter["layers"], auxiliary = False,drop_prob = hyperparameter["p"], genotype = gen, binary = binary)
      model = model.cuda(device = cuda_device)
      """
      ### Train the model
      """
      #pretrain_naswot = evaluator.score_naswot(model,testloader)
      if multibatch:
        train_model_multibatch(model , hyperparameter, trainloader , hyperparameter["epochs"], batch_size , cuda_device, augment_num = 1, graph = plot_queue, binary = binary) 
      else:  
        train_model_aug(model , hyperparameter, trainloader , hyperparameter["epochs"], batch_size , cuda_device, augment_num = 1, graph = plot_queue, binary = binary) 
      """
      ### Test the model
      """
      evaluator.forward_pass(model, testloader_full,binary)
      evaluator.predictions(model_is_binary = binary , THRESHOLD = THRESHOLD)

      ### Get Metrics
      total = evaluator.T()
      acc  =  evaluator.T_ACC()
      recall = evaluator.TPR(1)
      recall_total = evaluator.P(1)
      sup = evaluator.sup_loss(model, testloader)
      sup10 = evaluator.sup_loss(model, testloader, n_augment = 10)
      unsup10 = evaluator.unsup_loss(model, testloader, n_augment = 10)
      unsup = evaluator.unsup_loss(model, testloader)
      #cons = evaluator.c_loss(model, testloader)
      #cons_10 = evaluator.c_loss(model, testloader,10)
      #posttrain_naswot = evaluator.score_naswot(model,testloader)
      #df_loss = evaluator.loss_over_sample_size(model, dataset_test)
      print("Supervised Loss: {} -- Unsupvised Loss: {}".format(sup,unsup))
      print("Supervised Loss(10): {} -- Unsupervised Loss(10): {}".format(sup10,unsup10))
      #print("NASWOT PRE: {} -- NASWOT POST: {}".format(pretrain_naswot,posttrain_naswot))
      print("Accuracy: ", "%.4f" % ((acc)*100), "%")
      print("Recall: ", "%.4f" % ((recall)*100), "%")

      ### Save Model
      def save_model(model, hyperparameter):
        model_zoo = "{}/scripts/model_zoo/".format(os.environ["HOME"])
        torch.save(model.state_dict() , model_zoo+"-Acc-{}-Rec-{}".format(acc, recall))
        save_obj( hyperparameter , model_zoo+"hps/"+"-Acc-{}-Rec-{}".format(acc , recall) )


  print("Final Scores -- ACC: {} -- REC: {}".format(acc, recall))
  return acc, recall, sup.item(), unsup.item(), sup10.item(),unsup10.item()

if __name__ == "__main__":
  import multiprocessing
  hyperparameter = {'channels': 128, 'crop': 0.8435749854867439, 'crop_rate': 1.0789375913465558, 'cut_mix': 0.3606216485561041, 'cut_mix_rate': 0.006043016512336742, 'cut_out': 0.3759927924492086, 'cut_out_rate': 0.665600887946834, 'epochs': 80, 'jitter': 0.2565783789758802, 'jitter_rate': 1.4762754789497605, 'layers': 1, 'lr': 0.0016466283692574232, 'mix_up': 0.609992555365736, 'mix_up_rate': 0.36211335694823126, 'normal_index_0_0': 0, 'normal_index_0_1': 1, 'normal_index_1_0': 0, 'normal_index_1_1': 2, 'normal_index_2_0': 1, 'normal_index_2_1': 3, 'normal_index_3_0': 3, 'normal_index_3_1': 3, 'normal_node_0_0': 'avg_pool_3x3', 'normal_node_0_1': 'sep_conv_5x5', 'normal_node_1_0': 'dil_conv_3x3', 'normal_node_1_1': 'skip_connect', 'normal_node_2_0': 'max_pool_3x3', 'normal_node_2_1': 'avg_pool_3x3', 'normal_node_3_0': 'avg_pool_3x3', 'normal_node_3_1': 'sep_conv_5x5', 'p': 0.14391084865105958, 'reduction_index_0_0': 0, 'reduction_index_0_1': 1, 'reduction_index_1_0': 0, 'reduction_index_1_1': 0, 'reduction_index_2_0': 2, 'reduction_index_2_1': 2, 'reduction_index_3_0': 2, 'reduction_index_3_1': 3, 'reduction_node_0_0': 'sep_conv_3x3', 'reduction_node_0_1': 'sep_conv_7x7', 'reduction_node_1_0': 'dil_conv_3x3', 'reduction_node_1_1': 'skip_connect', 'reduction_node_2_0': 'skip_connect', 'reduction_node_2_1': 'avg_pool_3x3', 'reduction_node_3_0': 'dil_conv_3x3', 'reduction_node_3_1': 'sep_conv_7x7', 'scaling': 0.4499911429155192, 'scaling_rate': 1.0704794966200513, 'window_warp_num': 5, 'window_warp_rate': 1.8569110504223219}

  #0.8125,0.6896551724137931
  hyperparameter = {'normal_index_0_0': 1, 'normal_index_0_1': 0, 'normal_index_1_0': 1, 'normal_index_1_1': 0, 'normal_index_2_0': 3, 'normal_index_2_1': 2, 'normal_index_3_0': 0, 'normal_index_3_1': 3, 'normal_node_0_0': 'skip_connect', 'normal_node_0_1': 'dil_conv_5x5', 'normal_node_1_0': 'max_pool_3x3', 'normal_node_1_1': 'max_pool_3x3', 'normal_node_2_0': 'sep_conv_7x7', 'normal_node_2_1': 'none', 'normal_node_3_0': 'avg_pool_3x3', 'normal_node_3_1': 'sep_conv_5x5', 'reduction_index_0_0': 1, 'reduction_index_0_1': 0, 'reduction_index_1_0': 0, 'reduction_index_1_1': 2, 'reduction_index_2_0': 1, 'reduction_index_2_1': 3, 'reduction_index_3_0': 2, 'reduction_index_3_1': 4, 'reduction_node_0_0': 'none', 'reduction_node_0_1': 'avg_pool_3x3', 'reduction_node_1_0': 'max_pool_3x3', 'reduction_node_1_1': 'none', 'reduction_node_2_0': 'max_pool_3x3', 'reduction_node_2_1': 'sep_conv_5x5', 'reduction_node_3_0': 'dil_conv_5x5', 'reduction_node_3_1': 'avg_pool_3x3', 'batch_size': 2, 'channels': 27, 'jitter': 0.1241258424762939, 'jitter_rate': 0.5439942968995378, 'mix_up': 0.19412584247629389, 'mix_up_rate': 0.5439942968995378, 'cut_mix': 0.19412584247629389, 'cut_mix_rate': 0.5439942968995378, 'cut_out': 0.0941258424762939, 'cut_out_rate': 0.7439942968995378, 'crop': 0.19412584247629389, 'crop_rate': 0.5439942968995378, 'scaling': 0.001317169415702424, 'scaling_rate': 0.4353430973459786, 'window_warp_num': 3, 'window_warp_rate': 1.4001548161604196, 'lr': 0.005170869707739693, 'p': 0.00296905723528657, 'epochs': 70, 'layers': 3}

  hyperparameter = {'normal_index_0_0': 0, 'normal_index_0_1': 0, 'normal_index_1_0': 1, 'normal_index_1_1': 2, 'normal_index_2_0': 0, 'normal_index_2_1': 3, 'normal_index_3_0': 2, 'normal_index_3_1': 2, 'normal_node_0_0': 'avg_pool_3x3', 'normal_node_0_1': 'none', 'normal_node_1_0': 'max_pool_3x3', 'normal_node_1_1': 'skip_connect', 'normal_node_2_0': 'sep_conv_7x7', 'normal_node_2_1': 'dil_conv_3x3', 'normal_node_3_0': 'skip_connect', 'normal_node_3_1': 'dil_conv_3x3', 'reduction_index_0_0': 0, 'reduction_index_0_1': 0, 'reduction_index_1_0': 2, 'reduction_index_1_1': 1, 'reduction_index_2_0': 3, 'reduction_index_2_1': 0, 'reduction_index_3_0': 1, 'reduction_index_3_1': 4, 'reduction_node_0_0': 'sep_conv_3x3', 'reduction_node_0_1': 'sep_conv_5x5', 'reduction_node_1_0': 'sep_conv_7x7', 'reduction_node_1_1': 'skip_connect', 'reduction_node_2_0': 'max_pool_3x3', 'reduction_node_2_1': 'skip_connect', 'reduction_node_3_0': 'max_pool_3x3', 'reduction_node_3_1': 'dil_conv_5x5', 'batch_size': 2, 'channels': 27, 'jitter': 0.1241258424762939, 'jitter_rate': 0.5439942968995378, 'mix_up': 0.19412584247629389, 'mix_up_rate': 0.5439942968995378, 'cut_mix': 0.19412584247629389, 'cut_mix_rate': 0.5439942968995378, 'cut_out': 0.0941258424762939, 'cut_out_rate': 0.7439942968995378, 'crop': 0.19412584247629389, 'crop_rate': 0.5439942968995378, 'scaling': 0.001317169415702424, 'scaling_rate': 0.4353430973459786, 'window_warp_num': 3, 'window_warp_rate': 1.4001548161604196, 'lr': 0.005170869707739693, 'p': 0.00296905723528657, 'epochs': 70, 'layers': 3}
  hpo = {'batch_size': 2, 'channels': 64, 'jitter': 0.01241258424762939, 'jitter_rate': 0.5439942968995378, 'mix_up': 0.19412584247629389, 'mix_up_rate': 0.5439942968995378, 'cut_mix': 0.19412584247629389, 'cut_mix_rate': 0.5439942968995378, 'cut_out': 0.0941258424762939, 'cut_out_rate': 0.7439942968995378, 'crop': 0.19412584247629389, 'crop_rate': 0.5439942968995378, 'scaling': 0.001317169415702424, 'scaling_rate': 0.4353430973459786, 'window_warp_num': 3, 'window_warp_rate': 1.4001548161604196, 'lr': 0.0025170869707739693, 'p': 0.00, 'epochs': 1, 'layers': 3}
  #0.8125,0.7931034482758621,"
  hyperparameter = {'normal_index_0_0': 0, 'normal_index_0_1': 0, 'normal_index_1_0': 2, 'normal_index_1_1': 0, 'normal_index_2_0': 0, 'normal_index_2_1': 0, 'normal_index_3_0': 2, 'normal_index_3_1': 2, 'normal_node_0_0': 'avg_pool_3x3', 'normal_node_0_1': 'none', 'normal_node_1_0': 'skip_connect', 'normal_node_1_1': 'max_pool_3x3', 'normal_node_2_0': 'sep_conv_5x5', 'normal_node_2_1': 'none', 'normal_node_3_0': 'avg_pool_3x3', 'normal_node_3_1': 'dil_conv_3x3', 'reduction_index_0_0': 0, 'reduction_index_0_1': 1, 'reduction_index_1_0': 2, 'reduction_index_1_1': 0, 'reduction_index_2_0': 3, 'reduction_index_2_1': 1, 'reduction_index_3_0': 1, 'reduction_index_3_1': 2, 'reduction_node_0_0': 'skip_connect', 'reduction_node_0_1': 'none', 'reduction_node_1_0': 'max_pool_3x3', 'reduction_node_1_1': 'avg_pool_3x3', 'reduction_node_2_0': 'skip_connect', 'reduction_node_2_1': 'sep_conv_5x5', 'reduction_node_3_0': 'sep_conv_5x5', 'reduction_node_3_1': 'sep_conv_3x3'}

  #0.8170731707317073,0.6486486486486487,"
  hyperparameter = {'normal_index_0_0': 0, 'normal_index_0_1': 0, 'normal_index_1_0': 1, 'normal_index_1_1': 1, 'normal_index_2_0': 3, 'normal_index_2_1': 3, 'normal_index_3_0': 2, 'normal_index_3_1': 2, 'normal_node_0_0': 'avg_pool_3x3', 'normal_node_0_1': 'sep_conv_7x7', 'normal_node_1_0': 'sep_conv_5x5', 'normal_node_1_1': 'sep_conv_5x5', 'normal_node_2_0': 'sep_conv_7x7', 'normal_node_2_1': 'avg_pool_3x3', 'normal_node_3_0': 'skip_connect', 'normal_node_3_1': 'sep_conv_5x5', 'reduction_index_0_0': 1, 'reduction_index_0_1': 0, 'reduction_index_1_0': 0, 'reduction_index_1_1': 1, 'reduction_index_2_0': 1, 'reduction_index_2_1': 2, 'reduction_index_3_0': 4, 'reduction_index_3_1': 1, 'reduction_node_0_0': 'sep_conv_3x3', 'reduction_node_0_1': 'dil_conv_5x5', 'reduction_node_1_0': 'none', 'reduction_node_1_1': 'max_pool_3x3', 'reduction_node_2_0': 'dil_conv_3x3', 'reduction_node_2_1': 'avg_pool_3x3', 'reduction_node_3_0': 'none', 'reduction_node_3_1': 'sep_conv_5x5'}
  hyperparameter.update(hpo)
  _compute(hyperparameter, binary = False)
  exit()

