import numpy as np 
import time
import os 
import matplotlib.pyplot as plt
from HPO.utils.model import NetworkMain
from HPO.utils.DARTS_utils import config_space_2_DARTS
from HPO.utils.FCN import FCN 
import pandas as pd
import torch
from HPO.data.datasets import Test_repsol_full , Mixed_repsol_full, repsol_unlabeled
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
from HPO.utils.time_series_augmentation import permutation , magnitude_warp, time_warp
from HPO.utils.time_series_augmentation_torch import jitter, scaling, rotation
from HPO.utils.worker_train import train_model, collate_fn_padd, train_model_bt, collate_fn_padd_x, train_model_aug
from HPO.utils.weight_freezing import freeze_FCN, freeze_resnet
from HPO.utils.ResNet1d import resnet18
from HPO.utils.files import save_obj
from HPO.workers.repeat_worker import worker_wrapper, one_out_cv_aug, one_out_cv
from queue import Empty
from sklearn.model_selection import KFold
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
      for crashes in range(3):
      try:
        acc , rec =  _compute(hyperparameter = config , cuda_device = device)
      except:
        print("Model crash: {} ".format(crashes+1))
        time.sleep(60)
        if crashes == 2:
          print("Final Crash giving score of zero")
          acc , rec = 0 , 0 
      res.put([config , acc , rec ]) 

  torch.cuda.empty_cache()
   
def _compute(hyperparameter,budget = 4, in_model = None , train_dataset = None,  test_dataset = None, cuda_device = None,plot_queue = None, model_id = None, binary = True):
  ### Configuration 
  THRESHOLD = 0.4 #Cut off for classification
  batch_size = 2 
  OUTER_LOOP, INNER_FOLDS = random.choice([[2,4],[3,3]])

  if cuda_device == None:
     cuda_device = 0# torch.cuda.current_device()

  dataset = Mixed_repsol_full(0,augmentations_on = False)
  torch.cuda.set_device(cuda_device)

  print("Cuda Device Value: ", cuda_device)
   

  gen = config_space_2_DARTS(hyperparameter,reduction = False)
  n_classes = dataset.get_n_classes()
  kfold = KFold(n_splits = INNER_FOLDS, shuffle = True)
  evaluator = Evaluator(1, n_classes,cuda_device) 

  for outer in range(OUTER_LOOP):

    for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
      print('--Outer Loop-- {}---Fold No.--{}----------------------'.format(outer,fold))
      torch.cuda.empty_cache()
      train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
      test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
     
      trainloader = torch.utils.data.DataLoader(
                          dataset,collate_fn = collate_fn_padd, 
                          batch_size=batch_size, sampler=train_subsampler, drop_last = True)
      testloader = torch.utils.data.DataLoader(
                          dataset,collate_fn = collate_fn_padd,
                          batch_size=1, sampler=test_subsampler)

      ### Build Model
      model = NetworkMain(27,hyperparameter["channels"],num_classes= 2 , layers = hyperparameter["layers"], auxiliary = False,drop_prob = hyperparameter["p"], genotype = gen, binary = binary)
      model = model.cuda(device = cuda_device)
      """
      ### Train the model
      """
      train_model_aug(model , hyperparameter, trainloader , hyperparameter["epochs"], batch_size , cuda_device, augment_num = 1, graph = plot_queue, binary = binary) 
      """
      ### Test the model
      """
      evaluator.forward_pass(model, testloader,binary)
      evaluator.predictions(model_is_binary = binary , THRESHOLD = THRESHOLD)

      ### Get Metrics
      total = evaluator.T()
      acc  =  evaluator.T_ACC()
      recall = evaluator.TPR(1)
      recall_total = evaluator.P(1)

      print("Accuracy: ", "%.4f" % ((acc)*100), "%")
      print("Recall: ", "%.4f" % ((recall)*100), "%")

      ### Save Model
      def save_model(model, hyperparameter):
        model_zoo = "{}/scripts/model_zoo/".format(os.environ["HOME"])
        torch.save(model.state_dict() , model_zoo+"-Acc-{}-Rec-{}".format(acc, recall))
        save_obj( hyperparameter , model_zoo+"hps/"+"-Acc-{}-Rec-{}".format(acc , recall) )

      save_model(model,hyperparameter)

  print("Final Scores -- ACC: {} -- REC: {}".format(acc, recall))
  return acc, recall

if __name__ == "__main__":
  import multiprocessing
  user = input("Overwrite old DataFrame? (y/n)")
  if user.lower() == "y":
    df = pd.DataFrame(columns = ["Sample_ID","Source" , "Label", "Prediction_0", "Prediction_1","Output", "Correct", "Model_ID"])
    df.to_csv("results_df.csv")
    print("New DataFrame Created")
  for i in range(10):
    hyperparameter = {'T_0': 3,'T_mult':2, 'normal_cell_1_ops_8_input_1': 0, 'augmentations': 171, 'c1_weight': 1.2167576457622766, 'channels': 27, 'epochs': 50, 'layers': 4, 
      'lr': 0.0007072866653232726, 'normal_cell_1_num_ops': 1, 'normal_cell_1_ops_1_input_1': 0, 'normal_cell_1_ops_1_input_2': 0, 'normal_cell_1_ops_1_type': 'MaxPool5', 'normal_cell_1_ops_2_input_1': 1, 'normal_cell_1_ops_2_input_2': 0, 'normal_cell_1_ops_2_type': 'Conv5', 'normal_cell_1_ops_3_input_1': 2, 'normal_cell_1_ops_3_input_2': 0, 'normal_cell_1_ops_3_type': 'Conv7', 'normal_cell_1_ops_4_input_1': 0, 'normal_cell_1_ops_4_input_2': 0, 'normal_cell_1_ops_4_type': 'SepConv3', 'normal_cell_1_ops_5_input_1': 3, 'normal_cell_1_ops_5_input_2': 3, 'normal_cell_1_ops_5_type': 'AvgPool7', 'normal_cell_1_ops_6_input_1': 1, 'normal_cell_1_ops_6_input_2': 0, 'normal_cell_1_ops_6_type': 'MaxPool5', 'normal_cell_1_ops_7_input_1': 0, 'normal_cell_1_ops_7_input_2': 2, 'normal_cell_1_ops_7_type':     'SepConv5', 'normal_cell_1_ops_8_input_2': 6, 'normal_cell_1_ops_8_type': 'Conv3', 'normal_cell_1_ops_9_input_1': 4, 'normal_cell_1_ops_9_input_2': 1, 'normal_cell_1_ops_9_type': 'Conv7', 'num_conv': 1, 'num_re': 1, 'p': 0.02479858526104134, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 'reduction_cell_1_ops_1_type': 'FactorizedReduce'}
    #0.7790697674418605,0.5853658536585366
    hyperparameter  ={'T_0': 20, 'T_mult': 1, 'batch_size': 2, 'channels': 27, 'epochs': 155, 'layers': 8, 'lr': 0.0019989653577959284, 'normal_index_0_0': 1, 'normal_index_0_1': 1, 'normal_index_1_0': 1, 'normal_index_1_1': 1, 'normal_index_2_0': 3, 'normal_index_2_1': 3, 'normal_index_3_0': 4, 'normal_index_3_1': 1, 'normal_node_0_0': 'skip_connect', 'normal_node_0_1': 'dil_conv_3x3', 'normal_node_1_0': 'sep_conv_3x3', 'normal_node_1_1': 'dil_conv_5x5', 'normal_node_2_0': 'max_pool_3x3', 'normal_node_2_1': 'sep_conv_3x3', 'normal_node_3_0': 'dil_conv_3x3', 'normal_node_3_1': 'avg_pool_3x3', 'p': 0.17718387446598843, 'reduction_index_0_0': 0, 'reduction_index_0_1': 1, 'reduction_index_1_0': 0, 'reduction_index_1_1': 1, 'reduction_index_2_0': 1, 'reduction_index_2_1': 1, 'reduction_index_3_0': 4, 'reduction_index_3_1': 4, 'reduction_node_0_0': 'max_pool_3x3', 'reduction_node_0_1': 'dil_conv_5x5', 'reduction_node_1_0': 'skip_connect', 'reduction_node_1_1': 'max_pool_3x3', 'reduction_node_2_0': 'none', 'reduction_node_2_1': 'max_pool_3x3', 'reduction_node_3_0': 'sep_conv_3x3', 'reduction_node_3_1': 'avg_pool_3x3'}    
    hyperparameter = {
  "T_0" : 10,
  "c1" : 2.5,
  "T_mult" : 1,
  "batch_size" : 2,
  "channels" : 54,
  'jitter': 0.19412584247629389, 'jitter_rate': 0.5439942968995378, 'mix_up': 0.19412584247629389, 'mix_up_rate': 0.5439942968995378,
  'cut_mix': 0.19412584247629389, 'cut_mix_rate': 0.5439942968995378,'cut_out': 0.19412584247629389, 'cut_out_rate': 0.5439942968995378,
  'crop': 0.19412584247629389, 'crop_rate': 0.5439942968995378,
  'scaling': 0.0001317169415702424, 'scaling_rate': 0.23534309734597858, 'window_warp_num': 2, 'window_warp_rate': 2.40015481616041954,
  'lr': 0.0005170869707739693, 'p': 0.00296905723528657, 
  "epochs" : 98,
  "layers" : 3,
  "normal_index_0_0" : 0,
  "normal_index_0_1" : 1,
  "normal_index_1_0" : 1,
  "normal_index_1_1" : 1,
  "normal_index_2_0" : 3,
  "normal_index_2_1" : 3,
  "normal_index_3_0" : 4,
  "normal_index_3_1" : 1,
  "normal_node_0_0" : 'skip_connect',
  "normal_node_0_1" : 'dil_conv_3x3',
  "normal_node_1_0" : 'sep_conv_3x3',
  "normal_node_1_1" : 'sep_conv_15x15',
  "normal_node_2_0" : 'max_pool_3x3',
  "normal_node_2_1" : 'sep_conv_3x3',
  "normal_node_3_0" : 'dil_conv_3x3',
  "normal_node_3_1" : 'skip_connect',
  "p" : 0.012718387446598843,
  "reduction_index_0_0" : 0,
  "reduction_index_0_1" : 1,
  "reduction_index_1_0" : 0,
  "reduction_index_1_1" : 1,
  "reduction_index_2_0" : 3,
  "reduction_index_2_1" : 1,
  "reduction_index_3_0" : 3,
  "reduction_index_3_1" : 4,
  "reduction_node_0_0" : 'max_pool_3x3',
  "reduction_node_0_1" : 'avg_pool_3x3',
  "reduction_node_1_0" : 'skip_connect',
  "reduction_node_1_1" : 'skip_connect',
  "reduction_node_2_0" : 'max_pool_3x3',
  "reduction_node_2_1" : 'max_pool_3x3',
  "reduction_node_3_0" : 'sep_conv_3x3',
  "reduction_node_3_1" : 'avg_pool_3x3'}

    queue = multiprocessing.Queue()
    plotter = LivePlot(queue)
    plot_process = multiprocessing.Process(target=plotter.show,args=())
    plot_process.start()
    _compute(hyperparameter,model_id = "Binary_run_aug_intense_{}".format(i), binary = True, plot_queue = queue)
    plot_process.join()
    exit()

