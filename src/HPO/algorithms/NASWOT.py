
import math
import random
from HPO.utils.worker_helper import collate_fn_padd
import matplotlib.pyplot as plt
from HPO.utils.model import NetworkMain
from HPO.utils.DARTS_utils import config_space_2_DARTS
import torch
import numpy as np
import seaborn as sns




def attach_hooks(model, hook):
    ##Attach Activation Hooks
    x = repr(model)
    for i in model.cells:
        if "FactorizedReduce" not in repr(i.preprocess0) :
          for x in i.preprocess0.op:
            if repr(x) == "ReLU()":
                x.register_forward_hook(hook)
          for x in i.preprocess1.op:
            if repr(x) == "ReLU()":
                x.register_forward_hook(hook)
        #print("Activation Functions: {} {}".format(act1, act2))
    for j in i._ops:
      if "Conv" in repr(j) and "FactorizedReduce" not in repr(j):
        for j_i in j.op:
          if repr(j_i ) == "ReLU()":
            j_i.register_forward_hook(hook)




def counting_forward_hook(module, inp, out):
        global net_K
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        net_K = net_K + K.cpu().numpy() + K2.cpu().numpy()
    

class Model:
  def __init__(self, network, batch_size , N,device = None):
    self.network = network
    self.K = torch.zeros(batch_size, batch_size)
    self.N = N
    self.device = device
    self.score = 0
    if device != None:
      self.network = self.network.to(self.device)
      
       
  def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    self.score = ld
    return self.score

  def score(self, dataloader):
    for i in range(self.N):
      samples, labels = next(iter(dataloader))
      if self.device != None:
        samples = samples.to(self.device)
      self.network(samples)
      return self.hooklogdet(self.K) 

  def counting_forward_hook(self, module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        self.K = self.K + K.cpu().numpy() + K2.cpu().numpy()


class Res:
  def __init__(self,idx , configuration, score = None, accuracy = None):
    self.idx= idx
    self.configuration = configuration
    self.score = score
    self.accuracy = accuracy




def main(worker , configspace, dataset):
  ###Configuration
  BATCH_SIZE = 128
  N = 16
  SEARCH_SIZE = 500
  trainloader = torch.utils.data.DataLoader(
                    dataset,collate_fn = collate_fn_padd, 
                    batch_size=BATCH_SIZE, drop_last = True)
  result_list = []
  
  for iteration in range(SEARCH_SIZE):
    configuration = configspace.sample_configuration()
    results = Res(iteration,configuration)
    genotype = config_space_2_DARTS(configuration)
    network = NetworkMain(num_features,configuration["channels"],2,3,False,0,genotype) ##Initialise Network
    model = Model(network, BATCH_SIZE, N , device = 0)
    attach_hooks(model.network,model.counting_forward_hook)   
    results.score = model.score(trainloader)
    results_list.append(results)

  scores = []
  configs = []
  idx = []
  for i in results_list:
    scores.append(i.score)
    idx.append(i.idx)
    configs.append(i.configuration)
    
  



