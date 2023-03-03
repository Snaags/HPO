from HPO.utils.seed import set_seed
import torch

def __compute( ID, configs , gpus , res   , JSON_CONFIG, _compute):
  set_seed(JSON_CONFIG)
  device = None
  config = None
  print("Starting process: {}".format(ID))
  while not configs.empty():
    try:
      if device == None:
        device = gpus.get(timeout = 10)
      config = configs.get(timeout = 10)
      print("configs in queue: ",configs.qsize())
    except Empty:
      if device != None:
        gpus.put(device)
        return
      
    except:
      torch.cuda.empty_cache()
      if device != None:
        gpus.put(device)
      return
    if config != None:
      print("Got Configuration!")

    if device != None:
      print("Starting config with device: {}".format(device))
      acc , rec, params =  _compute(hyperparameter = config , cuda_device = device,JSON_CONFIG = JSON_CONFIG)
      res.put([config , acc , rec ,params]) 

  torch.cuda.empty_cache()
  print("Got out of configs.empty loop")
  return None


