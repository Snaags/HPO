import json
from HPO.utils.model_graph import ModelGraph
from HPO.searchspaces.graph_search_space import GraphConfigSpace
import numpy as np 
from HPO.data.UEA_datasets import UEA_Train, UEA_Test, UEA_Full
import time
import sys
from HPO.utils.utils import MetricLogger, BernoulliLogger, print_file
import os 
import matplotlib.pyplot as plt
from HPO.utils.model import NetworkMain
from HPO.utils.DARTS_utils import config_space_2_DARTS
from HPO.utils.FCN import FCN 
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from HPO.data.teps_datasets import Train_TEPS , Test_TEPS
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchsummary import summary
import random
import HPO.utils.augmentation as aug
from HPO.utils.train_utils import collate_fn_padd,BalancedBatchSampler, highest_power_of_two
from HPO.utils.train import train_model
from HPO.utils.weight_freezing import freeze_FCN, freeze_resnet
from HPO.utils.ResNet1d import resnet18
from HPO.utils.files import save_obj
from queue import Empty
from sklearn.model_selection import StratifiedKFold as KFold
from collections import namedtuple
from HPO.utils.worker_score import Evaluator 
from scipy.stats import loguniform 
from HPO.utils.worker_utils import LivePlot
from HPO.workers.worker_wrapper import __compute
from HPO.data.dataset import get_dataset
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')




def compute(*args, **kwargs):
  __compute(*args, **kwargs, _compute = _compute)
  return True


def _compute(hyperparameter,cuda_device, JSON_CONFIG, train_dataset, test_dataset):
  start = time.time()
  ### Configuration 
  if type(JSON_CONFIG) != dict:
    with open(JSON_CONFIG) as f:
      data = json.load(f)
  else:
    data = JSON_CONFIG
  dataset = train_dataset
  SETTINGS = data["WORKER_CONFIG"]
  SETTINGS["ID"] = hyperparameter["ID"]
  SETTINGS["database"] = data["DATABASE_NAME"]
  SETTINGS["experiment"] = data["EXPERIMENT_NAME"]
  ARCH_SETTINGS = data["ARCHITECTURE_CONFIG"]
  SAVE_PATH = data["SEARCH_CONFIG"]["PATH"]
  if "RANDOMISE_HYPERPARAMETERS" in SETTINGS:
    if SETTINGS["RANDOMISE_HYPERPARAMETERS"]:
      for i in SETTINGS["RANDOMISE_SETTINGS"]:
        if i == "EPOCHS":
          SETTINGS[i] = random.randint(SETTINGS["RANDOMISE_SETTINGS"][i][0],SETTINGS["RANDOMISE_SETTINGS"][i][1])
        else:
          SETTINGS[i] = loguniform.rvs(float(SETTINGS["RANDOMISE_SETTINGS"][i][1]),float(SETTINGS["RANDOMISE_SETTINGS"][i][0]))

  acc = []
  metric_logger = MetricLogger(SAVE_PATH) 
  binary_logger = BernoulliLogger(SAVE_PATH,hyperparameter["ID"]) 
  recall = []
  if cuda_device == None:
     cuda_device = 3
  torch.cuda.empty_cache()
  #print(hyperparameter)
  torch.cuda.set_device(cuda_device)
  #torch.autograd.set_detect_anomaly(True)
  #with torch.autograd.profiler.profile() as prof:

  ##Dataset Initialisation
  name = SETTINGS["DATASET_CONFIG"]["NAME"]
  if data["GENERATE_PARTITION"]:
    DS_PATH = SETTINGS["DATASET_CONFIG"]["DATASET_PATH"]
  else:
    DS_PATH = None
  if "AUGMENTATIONS" in SETTINGS:
    augs = aug.initialise_augmentations(SETTINGS["AUGMENTATIONS"])
  else: 
    augs = None

  ##CONFIGURE LOW DATA OR HIGH DATA CONFIGURATION
  if len(train_dataset) > 500:
    SETTINGS["REPEAT"] = 5
  else:
    SETTINGS["REPEAT"] = 5

  train_args = {"cuda_device":cuda_device,"augmentation" : augs, "binary" :SETTINGS["BINARY"],"path" : DS_PATH}
  test_args = {"cuda_device":cuda_device,"augmentation" :None, "binary" :SETTINGS["BINARY"],"path" : DS_PATH}

  n_classes = train_dataset.get_n_classes()
  multibatch = False
  torch.cuda.empty_cache()
   
  for _ in range(SETTINGS["REPEAT"]):
    
    if SETTINGS["GROUPED_RESAMPLES"]:
      # Initialize GroupKFold cross-validator with desired number of splits
      kfold = GroupKFold(n_splits=5, shuffle = True)
      splits = [(None,None)]*SETTINGS["RESAMPLES"]


    elif SETTINGS["RESAMPLES"]:
      if "MAX_REPEAT" in SETTINGS and SETTINGS["MAX_REPEAT"]:
        splits = dataset.min_samples_per_class()
      else:
        splits = min([dataset.min_samples_per_class(), 3])
      kfold = KFold(n_splits = splits, shuffle = True,random_state = _)
      splits = [(None,None)]*SETTINGS["RESAMPLES"]

    elif SETTINGS["CROSS_VALIDATION_FOLDS"] == False: 
      splits = [(None,None)]
    elif SETTINGS["CROSS_VALIDATION_FOLDS"]:
      kfold = KFold(n_splits = SETTINGS["CROSS_VALIDATION_FOLDS"], shuffle = True)
      splits = kfold.split(dataset.x.cpu().numpy(),y = dataset.y.cpu().numpy())

    
    for fold, (train_ids, test_ids) in enumerate(splits):    
      #print('---Fold No.--{}--------------------'.format(fold))
      torch.cuda.empty_cache()
      if SETTINGS["GROUPED_RESAMPLES"]:
         train_ids, test_ids = next(kfold.split(dataset.x.cpu().numpy(),y = dataset.y.cpu().numpy(),groups =dataset.groups))
         dataset.get_groups(train_ids,test_ids)
      elif SETTINGS["RESAMPLES"]:
        train_ids, test_ids = next(kfold.split(dataset.x.cpu().numpy(),y = dataset.y.cpu().numpy()))
        if SETTINGS["MAX_BATCH"]:
          SETTINGS["BATCH_SIZE"] = len(train_ids)
        else:
          SETTINGS["BATCH_SIZE"] = min( [highest_power_of_two(len(test_ids)),  SETTINGS["BATCH_SIZE"]]  )
      if SETTINGS["CROSS_VALIDATION_FOLDS"] or SETTINGS["RESAMPLES"]: 
        # Sample elements randomly from a given list of ids, no replacement.
      
        if SETTINGS["BALANCED_BATCH"]:
          train_batch_sampler = BalancedBatchSampler(dataset,SETTINGS["BATCH_SIZE"],train_ids)
          trainloader = torch.utils.data.DataLoader(dataset,collate_fn = collate_fn_padd,batch_sampler =train_batch_sampler)
        else: 
          train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
          
          trainloader = torch.utils.data.DataLoader(
                                  dataset,collate_fn = collate_fn_padd,sampler = train_subsampler,
                                  batch_size=SETTINGS["BATCH_SIZE"], drop_last = True)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        testloader = torch.utils.data.DataLoader(
                              dataset,collate_fn = collate_fn_padd,sampler = test_subsampler,
                              batch_size= len(test_ids) ,drop_last = True)
      else:
        trainloader = torch.utils.data.DataLoader(
                                train_dataset,collate_fn = collate_fn_padd,shuffle = True,
                                batch_size=SETTINGS["BATCH_SIZE"], drop_last = True)
        testloader = torch.utils.data.DataLoader(
                            test_dataset,collate_fn = collate_fn_padd,shuffle = True,
                            batch_size= SETTINGS["BATCH_SIZE"] ,drop_last = True)
      if SETTINGS["RESAMPLES"] or SETTINGS["CROSS_VALIDATION_FOLDS"]:
        dataset.enable_augmentation(augs)
      n_classes = test_dataset.get_n_classes()
      evaluator = Evaluator(len(test_ids), test_dataset.get_n_classes(),cuda_device,testloader = testloader)   
      #print("classes: {} - name: {}".format(train_dataset.get_n_classes(),name))
      #g = GraphConfigSpace(50)
      #s = g.sample_configuration()
      #s = s[0]
      if "stem" in hyperparameter["ops"]:
        stem_size = hyperparameter["ops"]["stem"]
      else:
        stem_size = ARCH_SETTINGS["STEM_SIZE"][0]
      model = ModelGraph(train_dataset.get_n_features(),stem_size,train_dataset.get_n_classes(),
          train_dataset.get_length(),hyperparameter["graph"],hyperparameter["ops"],device = cuda_device,
          binary = SETTINGS["BINARY"],dropout = SETTINGS["DROPOUT"],droppath = SETTINGS["DROPPATH"],
          raw_stem = SETTINGS["RAW_STEM"],embedding = SETTINGS["EMBEDDING"])


      if SETTINGS["EFFICIENT_WEIGHTS"] and "parent" in hyperparameter["ops"]:
          #print("LOADING PARENT ID", hyperparameter["ops"]["parent"])
          files = os.listdir("{}/weights/".format(SAVE_PATH))
          for i in files:
            splits = i.split("-")

            if ( int(splits[0]) == hyperparameter["ops"]["parent"]) and (int(splits[1]) == _):
              state_dict = torch.load("{}/weights/{}".format(SAVE_PATH,i))
              break
          else:
            print("Not found: {}".format(hyperparameter["ops"]["parent"]))
            print_file("{}/log.txt".format(SAVE_PATH),"[{}] Not found: {}".format(time.time(),hyperparameter["ops"]["parent"]))

          own_state = model.state_dict()
          for name, param in state_dict.items():
              if name not in own_state:
                   #print('Ignoring {} since it is not in current model.'.format(name))
                   continue
              if isinstance(param, nn.Parameter):
                  # backwards compatibility for serialized parameters
                  param = param.data
              try:
                  own_state[name].copy_(param)
                  #print('Successfully loaded {}'.format(name))
              except Exception:
                  pass
                  #print('While copying the parameter named {}, whose dimensions in the model are {} and dimensions in the saved model are {}, ...'.format(name, own_state[name].size(), param.size()))

          #print('Finished loading weights.')
      else:
          if SETTINGS["EPOCHS_INITIAL"]:
            SETTINGS["EPOCHS"] = SETTINGS["EPOCHS_INITIAL"]
      """
      if "parent" in hyperparameter["ops"]:
        print("LOADING PARENT ID", hyperparameter["ops"]["parent"])
        files = os.listdir("{}/weights/".format(SAVE_PATH))
        for i in files:
          splits = i.split("-")

          if ( int(splits[0]) == hyperparameter["ops"]["parent"]) and (int(splits[1]) == _):
            state_dict = torch.load("{}/weights/{}".format(SAVE_PATH,i))
            break
        print(model.load_state_dict(state_dict, strict=False))
      else:
        print("WARNING PARENT NOT IN OPS")
      """

      model = model.cuda(device = cuda_device)

      if SETTINGS["COMPILE"]:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)


      params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      #print("Size: {}".format(params))
      #summary(model, ( train_dataset.get_n_features(), train_dataset.get_length()))
      train_model(model , SETTINGS, trainloader , cuda_device, evaluator = evaluator if SETTINGS["LIVE_EVAL"] else None, fold = fold, repeat = _) 
    #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
      torch.cuda.empty_cache()
      model.eval()
      if (SETTINGS["RESAMPLES"] or SETTINGS["CROSS_VALIDATION_FOLDS"] ) and SETTINGS["TEST_TIME_AUGMENTATION"] == False:
        dataset.disable_augmentation()  
      evaluator.forward_pass(model, testloader,SETTINGS["BINARY"],n_iter = SETTINGS["TEST_ITERATION"])
      evaluator.predictions(model_is_binary = SETTINGS["BINARY"] , THRESHOLD = SETTINGS["THRESHOLD"],no_print = SETTINGS["LIVE_EVAL"])
      total = evaluator.T()
      acc.append( evaluator.T_ACC())
      recall.append(evaluator.TPR(1))
      recall_total = evaluator.P(1)
      #print("Accuracy: ", "%.4f" % ((acc[-1])*100), "%")
      #print("Recall: ", "%.4f" % ((recall[-1])*100), "%")
      if SETTINGS["SAVE_WEIGHTS"]:
        dp = 2
        #compare_weights_debug(model.state_dict(),,"{}/weights/{}-{}-{:.02f}".format(SAVE_PATH,hyperparameter["ID"],_,acc[-1]),hyperparameter["ID"])
        _p = "{}/weights/{}-{}-{:.0"+str(dp)+"f}"
        while os.path.exists(_p.format(SAVE_PATH,hyperparameter["ID"],_,acc[-1])):
          dp += 1
          _p = "{}/weights/{}-{}-{:.0"+str(dp)+"f}"
        torch.save(model.state_dict(),_p.format(SAVE_PATH,hyperparameter["ID"],_,acc[-1]))

      metric_logger.update({"ID" : hyperparameter["ID"], "accuracy" : acc[-1], "recall": recall[-1]})
      if False:
        binary_logger.update(evaluator.correct)
    acc_ = np.mean(acc)
    recall_ = np.mean(recall)
    #print("Average Accuracy: ", "%.4f" % ((acc_)*100), "%")
  return acc_, recall_,params



if __name__ == "__main__":
    from HPO.general_utils import load
    with open(sys.argv[1]) as f:
      DATA = json.load(f)
      HP = DATA["WORKER_CONFIG"]
      j = DATA
      j["WORKER_CONFIG"]["MODEL_VALIDATION_RATE"] = 5
      j["WORKER_CONFIG"]["REPEAT"] = 10
      j["WORKER_CONFIG"]["GROUPED_RESAMPLES"] = False
      j["WORKER_CONFIG"]["WEIGHT_AVERAGING_RATE"] =  False

      #j["WORKER_CONFIG"]["LR_MIN"] =  1e-07
      j["WORKER_CONFIG"]["RESAMPLES"] = False
      j["WORKER_CONFIG"]["EPOCHS"] = 50
      j["WORKER_CONFIG"]["PRINT_RATE_TRAIN"] = 50
      j["WORKER_CONFIG"]["LIVE_EVAL"] = True
      j["WORKER_CONFIG"]["EFFICIENT_WEIGHTS"] = False
      #j["WORKER_CONFIG"]["DATASET_CONFIG"]["NAME"] = "{}_Retrain".format(HP["DATASET_CONFIG"]["NAME"] )
      search = load( "{}/{}".format(DATA["SEARCH_CONFIG"]["PATH"],"evaluations.csv"))
      HP["ID"] = "val"
      HP["graph"] = search["config"][search["best"].index(min(search["best"]))]["graph"]
      HP["ops"] = search["config"][search["best"].index(min(search["best"]))]["ops"]
    exit()
    _compute(HP,2,j)
