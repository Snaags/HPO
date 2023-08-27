import json
from HPO.utils.model_graph import ModelGraph
from HPO.searchspaces.graph_search_space import GraphConfigSpace
import numpy as np 
from HPO.data.UEA_datasets import UEA_Train, UEA_Test, UEA_Full
import time
import sys
import torch.nn as nn
from HPO.utils.utils import MetricLogger, BernoulliLogger, print_file
import os 
import matplotlib.pyplot as plt
from HPO.utils.model import NetworkMain
from HPO.utils.DARTS_utils import config_space_2_DARTS
from HPO.utils.FCN import FCN 
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from HPO.data.teps_datasets import Train_TEPS , Test_TEPS
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchsummary import summary
import random
import HPO.utils.augmentation as aug
from HPO.utils.train_utils import collate_fn_padd,BalancedBatchSampler, highest_power_of_two, get_batch_size_from_n_batches
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
  #print(hyperparameter["hyperparameter"])
  #if "parent" in hyperparameter["ops"]:
  #  print("Starting: {} Child of {}".format(hyperparameter["ops"]["parent"],hyperparameter["ID"]))
  acc = []
  metric_logger = MetricLogger(SAVE_PATH) 
  binary_logger = BernoulliLogger(SAVE_PATH,hyperparameter["ID"]) 
  recall = []

  torch.cuda.empty_cache()
  torch.cuda.set_device(cuda_device)

  if "AUGMENTATIONS" in SETTINGS:
    augs = aug.initialise_augmentations(hyperparameter["hyperparameter"]["AUGMENTATIONS"])
  else:
    augs = None
   
  for _ in range(SETTINGS["REPEAT"]):
    


    if SETTINGS["RESAMPLES"]:
      if "MAX_REPEAT" in SETTINGS and SETTINGS["MAX_REPEAT"]:
        splits = dataset.min_samples_per_class()
      else:
        splits = min([dataset.min_samples_per_class(), 5])
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
         gss = GroupShuffleSplit(n_splits = 1 ,test_size=0.2,random_state = _)
         train_ids , test_ids  = next(gss.split(dataset.x, dataset.y, dataset.groups))
         dataset.get_groups(train_ids,test_ids)
      elif SETTINGS["RESAMPLES"]:
        train_ids, test_ids = next(kfold.split(dataset.x.cpu().numpy(),y = dataset.y.cpu().numpy()))



      SETTINGS["BATCH_SIZE"] = min( [highest_power_of_two(len(train_ids)),  get_batch_size_from_n_batches(hyperparameter["hyperparameter"]["BATCH_SIZE"],len(train_ids))]  )
      if SETTINGS["CROSS_VALIDATION_FOLDS"] or SETTINGS["RESAMPLES"]: 
        # Sample elements randomly from a given list of ids, no replacement.
      

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        
        trainloader = torch.utils.data.DataLoader(
                                dataset,collate_fn = collate_fn_padd,sampler = train_subsampler,
                                batch_size=SETTINGS["BATCH_SIZE"], drop_last = True)

        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        testloader = torch.utils.data.DataLoader(
                              dataset,collate_fn = collate_fn_padd,sampler = test_subsampler,
                              batch_size= 32,drop_last = False)


      if SETTINGS["RESAMPLES"] or SETTINGS["CROSS_VALIDATION_FOLDS"]:
        dataset.enable_augmentation(augs,train_ids)
      evaluator = Evaluator(32, test_dataset.get_n_classes(),cuda_device,testloader = testloader)   


      if "stem" in hyperparameter["ops"]:
        stem_size = hyperparameter["ops"]["stem"]
      else:
        stem_size = ARCH_SETTINGS["STEM_SIZE"][0]
      model = ModelGraph(train_dataset.get_n_features(),stem_size,train_dataset.get_n_classes(),
          train_dataset.get_length(),hyperparameter["graph"],hyperparameter["ops"],device = cuda_device,
          binary = SETTINGS["BINARY"],dropout = hyperparameter["hyperparameter"]["DROPOUT"],droppath = SETTINGS["DROPPATH"],
          raw_stem = SETTINGS["RAW_STEM"],embedding = SETTINGS["EMBEDDING"],auxiliary_head = True)


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
          errors = 0
          for name, param in state_dict.items():
              if name not in own_state:
                   #print('Ignoring {} since it is not in current model.'.format(name))
                   continue
              if isinstance(param, nn.Parameter):
                  # backwards compatibility for serialized parameters
                  param = param.data
                  param = (param * SETTINGS["LAMDBA"]) + torch.randn_like(param) * SETTINGS["GAMMA"] 
              try:
                  own_state[name].copy_(param)
                  #print('Successfully loaded {}'.format(name))
              except Exception:
                  pass
                  errors += 1 
                  #print('While copying the parameter named {}, whose dimensions in the model are {} and dimensions in the saved model are {}, ...'.format(name, own_state[name].size(), param.size()))
          #print("While copying parameters {} failed".format(errors))
          #print('Finished loading weights.')
      else:
          if SETTINGS["EPOCHS_INITIAL"]:
            hyperparameter["hyperparameter"]["EPOCHS"] = SETTINGS["EPOCHS_INITIAL"]
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
      hyperparameter["hyperparameter"]["SCHEDULE"] = False
      train_model(model , hyperparameter["hyperparameter"], trainloader , cuda_device,SETTINGS = SETTINGS, evaluator = evaluator if SETTINGS["LIVE_EVAL"] else None, fold = fold, repeat = _) 
      torch.cuda.empty_cache()
      model.eval()

      if (SETTINGS["RESAMPLES"] or SETTINGS["CROSS_VALIDATION_FOLDS"] ) and SETTINGS["TEST_TIME_AUGMENTATION"] == False:
        dataset.disable_augmentation()
      elif SETTINGS["TEST_TIME_AUGMENTATION"]:
        augs = aug.initialise_augmentations(SETTINGS["AUGMENTATIONS"])
        dataset.enable_augmentation(augs)


      evaluator.forward_pass(model, testloader,SETTINGS["BINARY"],n_iter = SETTINGS["TEST_ITERATION"])
      evaluator.predictions(model_is_binary = SETTINGS["BINARY"] , THRESHOLD = SETTINGS["THRESHOLD"],no_print = not SETTINGS["LIVE_EVAL"])
      total = evaluator.T()
      if "BALANCED_ACC" in SETTINGS and SETTINGS["BALANCED_ACC"]:
        aux_criterion = nn.MSELoss().cuda(device = cuda_device)
        loss = min([evaluator.calculate_loss(aux_criterion).item(),980])
        acc.append(evaluator.balanced_acc())
      else: 
        acc.append( evaluator.T_ACC())

      recall.append(evaluator.balanced_acc())
      recall_total = evaluator.P(1)

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
      j["WORKER_CONFIG"]["EFFICIENT_WEIGHTS"] = True
      #j["WORKER_CONFIG"]["DATASET_CONFIG"]["NAME"] = "{}_Retrain".format(HP["DATASET_CONFIG"]["NAME"] )
      search = load( "{}/{}".format(DATA["SEARCH_CONFIG"]["PATH"],"evaluations.csv"))
      HP["ID"] = "val"
      HP["graph"] = search["config"][search["best"].index(min(search["best"]))]["graph"]
      HP["ops"] = search["config"][search["best"].index(min(search["best"]))]["ops"]
    _compute(HP,0,j)
