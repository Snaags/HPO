import math
import random
from HPO.utils.model import NetworkMain
import torch
import numpy as np
import seaborn as sns
import csv

net_K = 0
def load_csv(file):

  acc_l = []
  rec_l = []
  config_l = []
  with open(file, "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      acc = float(row[0])
      rec = float(row[1])
      config = eval(row[2])
      acc_l.append(acc)
      rec_l.append(rec)
      config_l.append(config)

  return acc_l, rec_l , config_l

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach()

class naswot:
  def __init__(self,model, loader,N, device):
    self.model = model
    self.loader = loader
    self.N = N
    self.device = device
    ##Attach Activation Hooks
    for i in model.cells:
        if "FactorizedReduce" not in repr(i.preprocess0) :
          for x in i.preprocess0.op:
            if repr(x) == "ReLU()":
                x.register_backward_hook(self.counting_backward_hook)
                x.register_forward_hook(self.counting_forward_hook)
          for x in i.preprocess1.op:
            if repr(x) == "ReLU()":
                x.register_backward_hook(self.counting_backward_hook)
                x.register_forward_hook(self.counting_forward_hook)
        #print("Activation Functions: {} {}".format(act1, act2))
        for j in i._ops:
          if "Conv" in repr(j) and "FactorizedReduce" not in repr(j):
            for j_i in j.op:
              if repr(j_i ) == "ReLU()":
                j_i.register_backward_hook(self.counting_backward_hook)
                j_i.register_forward_hook(self.counting_forward_hook)

  def score(self):
    self.net_K = torch.zeros(self.N,self.N)
    for x,target in self.loader:
      x ,target = x.to(self.device).float(), target.to(self.device)
      self.model(x)
    print("naswot kernel : {}".format(self.net_K))
    return hooklogdet(self.net_K)
  def counting_forward_hook(self,module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        self.net_K = self.net_K + K.cpu().numpy() + K2.cpu().numpy()
    
  def counting_backward_hook(self,module, inp, out):
      module.visited_backwards = True
    

##Hooks from author implimentation
def counting_forward_hook(module, inp, out):
        global net_K
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        net_K = net_K + K.cpu().numpy() + K2.cpu().numpy()
    
def counting_backward_hook(module, inp, out):
    module.visited_backwards = True

def main():
  acc, rec, conf = load_csv("/home/snaags/uist/RegEvo.csv")
  acc_list = []
  dataset = Train_TEPS_split()
  N = 256
  N_iter = 8
  trainloader = torch.utils.data.DataLoader(
                      dataset,collate_fn = collate_fn_padd, 
                      batch_size=N, drop_last = True)
  configspace = DARTS_config.init_config()
  ###Loop through random models in searchspace
  for _ in range(100):
      c = configspace.sample_configuration()
      print("Model: {}".format(_))
      gen = config_space_2_DARTS(c)
      model = NetworkMain(52,c["channels"],2,3,False,0,gen) ##Initialise Network
      model.cuda()
  
      ##Attach Activation Hooks
      x = repr(model)
      for i in model.cells:
          if "FactorizedReduce" not in repr(i.preprocess0) :
            for x in i.preprocess0.op:
              if repr(x) == "ReLU()":
                  x.register_backward_hook(counting_backward_hook)
                  x.register_forward_hook(counting_forward_hook)
            for x in i.preprocess1.op:
              if repr(x) == "ReLU()":
                  x.register_backward_hook(counting_backward_hook)
                  x.register_forward_hook(counting_forward_hook)
          #print("Activation Functions: {} {}".format(act1, act2))
      for j in i._ops:
        if "Conv" in repr(j) and "FactorizedReduce" not in repr(j):
          for j_i in j.op:
            if repr(j_i ) == "ReLU()":
              j_i.register_backward_hook(counting_backward_hook)
              j_i.register_forward_hook(counting_forward_hook)
  
      net_K = torch.zeros(N,N)
  
      for i in range(N_iter):
        x, target = next(iter(trainloader))
        x ,target = x.to(device).float(), target.to(device)
        model(x)
      scores.append(hooklogdet(net_K))
      print(scores[-1])
  print(scores)
  print(acc)
  plt.plot(scores)
  plt.show()      


###Loop through random models in searchspace
def naswot_function(model,loader,N,device):
    ##Attach Activation Hooks
    for i in model.cells:
        if "FactorizedReduce" not in repr(i.preprocess0) :
          for x in i.preprocess0.op:
            if repr(x) == "ReLU()":
                x.register_backward_hook(counting_backward_hook)
                x.register_forward_hook(counting_forward_hook)
          for x in i.preprocess1.op:
            if repr(x) == "ReLU()":
                x.register_backward_hook(counting_backward_hook)
                x.register_forward_hook(counting_forward_hook)
        #print("Activation Functions: {} {}".format(act1, act2))
        for j in i._ops:
          if "Conv" in repr(j) and "FactorizedReduce" not in repr(j):
            for j_i in j.op:
              if repr(j_i ) == "ReLU()":
                j_i.register_backward_hook(counting_backward_hook)
                j_i.register_forward_hook(counting_forward_hook)

    net_K = torch.zeros(N,N)
    for x,target in loader:
      x ,target = x.to(device).float(), target.to(device)
      model(x)
    print("naswot kernel : {}".format(net_K))
    return hooklogdet(net_K)



