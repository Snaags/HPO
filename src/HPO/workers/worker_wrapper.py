from HPO.utils.seed import set_seed
import torch
from queue import Empty
from HPO.data.dataset import get_dataset
import HPO.utils.augmentation as aug
from sklearn.model_selection import StratifiedKFold as KFold
import json
import time
def __compute( ID, configs , gpus , res   , JSON_CONFIG, _compute):
  set_seed(JSON_CONFIG)
  device = None
  config = None
  print("Starting process: {}".format(ID))
  train, test = None, None
  WAIT = True
  while not configs.empty() or WAIT:
    if configs.empty():
      time.sleep(1)
      WAIT = False
      continue
    else:
      WAIT = True

    try:
      if device == None:
        device = gpus.get(timeout = 10)
      config = configs.get(timeout = 10)
    except Empty:
      if device != None:
        gpus.put(device)
        return
      
    except:
      torch.cuda.empty_cache()
      if device != None:
        gpus.put(device)
      return
    if device != None:
      if (train == None) and (test == None):
        train, test = load_dataset(JSON_CONFIG,device)
      elif device != train.x.get_device():
        train.update_device(device)
        test.update_device(device)
      train, test = load_dataset(JSON_CONFIG,device)
      acc , rec, params =  _compute(hyperparameter = config , cuda_device = device,JSON_CONFIG = JSON_CONFIG,train_dataset = train, test_dataset= test)
      res.put([config , acc , rec ,params]) 

  torch.cuda.empty_cache()
  print("Got out of configs.empty loop")
  return True



def load_dataset(JSON_CONFIG, cuda_device):
  if type(JSON_CONFIG) != dict:
      with open(JSON_CONFIG) as f:
        data = json.load(f)
  else:
      data = JSON_CONFIG
  SETTINGS = data["WORKER_CONFIG"]

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

  train_args = {"cuda_device":cuda_device,"augmentation" : augs, "binary" :SETTINGS["BINARY"],"path" : DS_PATH}
  test_args = {"cuda_device":cuda_device,"augmentation" :None, "binary" :SETTINGS["BINARY"],"path" : DS_PATH}

  if SETTINGS["RESAMPLES"]:
    dataset = get_dataset(name,train_args,None )
    kfold = KFold(n_splits = 2, shuffle = True)
    splits = [(None,None)]*SETTINGS["RESAMPLES"]
    train_dataset = dataset
    test_dataset = dataset
  elif SETTINGS["CROSS_VALIDATION_FOLDS"] == False: 
    train_dataset, test_dataset = get_dataset(name,train_args, test_args)
    splits = [(None,None)]
  elif SETTINGS["CROSS_VALIDATION_FOLDS"]:
    dataset, test_dataset = get_dataset(name,train_args, test_args)
    kfold = KFold(n_splits = SETTINGS["CROSS_VALIDATION_FOLDS"], shuffle = False)
    splits = kfold.split(dataset.x.cpu().numpy(),y = dataset.y.cpu().numpy())
    train_dataset = dataset
    test_dataset = dataset

  return train_dataset, test_dataset
