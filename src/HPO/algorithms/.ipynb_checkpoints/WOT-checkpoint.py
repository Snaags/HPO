import math
import matplotlib.pyplot as plt
from HPO.utils.model import NetworkMain
from HPO.utils.genotypes import NASNet
from HPO.utils.DARTS_utils import config_space_2_DARTS
import torch
import numpy as np
import seaborn as sns
from HPO.searchspaces import DARTS_config
import csv
hps_best = {'T_0': 3,
 'T_mult': 1,
 'batch_size': 2,
 'channels': 19,
 'epochs': 41,
 'layers': 3,
 'lr': 0.002993825743228492,
 'normal_index_0_0': 0,
 'normal_index_0_1': 1,
 'normal_index_1_0': 1,
 'normal_index_1_1': 1,
 'normal_index_2_0': 3,
 'normal_index_2_1': 3,
 'normal_index_3_0': 4,
 'normal_index_3_1': 1,
 'normal_node_0_0': 'skip_connect',
 'normal_node_0_1': 'dil_conv_3x3',
 'normal_node_1_0': 'sep_conv_3x3',
 'normal_node_1_1': 'dil_conv_5x5',
 'normal_node_2_0': 'max_pool_3x3',
 'normal_node_2_1': 'sep_conv_3x3',
 'normal_node_3_0': 'dil_conv_3x3',
 'normal_node_3_1': 'skip_connect',
 'p': 0.017718387446598843,
 'reduction_index_0_0': 0,
 'reduction_index_0_1': 1,
 'reduction_index_1_0': 0,
 'reduction_index_1_1': 1,
 'reduction_index_2_0': 3,
 'reduction_index_2_1': 1,
 'reduction_index_3_0': 3,
 'reduction_index_3_1': 4,
 'reduction_node_0_0': 'max_pool_3x3',
 'reduction_node_0_1': 'avg_pool_3x3',
 'reduction_node_1_0': 'skip_connect',
 'reduction_node_1_1': 'skip_connect',
 'reduction_node_2_0': 'max_pool_3x3',
 'reduction_node_2_1': 'max_pool_3x3',
 'reduction_node_3_0': 'sep_conv_3x3',
 'reduction_node_3_1': 'avg_pool_3x3'}

configspace = DARTS_config.init_config()

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

C = list()

def hamming_distance(c_1, c_2):
  
  x = torch.cdist(c_1,c_2, p = 0)
  return x

def hamming_kernel(C_list : list):
  K_h = torch.zeros(len(C_list), len(C_list))
  for i, c_i in enumerate(C_list):
    for j,c_j in enumerate(C_list):
      K_h[i,j] = len(c_j[0]) - hamming_distance(c_i,c_j)
  return K_h

def score(K_h):
  x = np.linalg.slogdet(K_h.detach().numpy())
  return x[1]



def activation_hook(inst, inp , out):
  hold = torch.reshape(out, (1, -1))
  C.append(hold)  


acc, rec, conf = load_csv("/home/snaags/uist/RegEvo.csv")
scores = []
for _, (c,a_acc) in enumerate(zip(conf,acc)):
  print("Model: {}".format(_))
  gen = config_space_2_DARTS(c)
  model = NetworkMain(16,4,2,3,False, 0,gen) ##Initialise Network
  
  ##Attach Activation Hooks
  x = repr(model)
  for i in model.cells:
    if "FactorizedReduce" not in repr(i.preprocess0) :
      for x in i.preprocess0.op:
        if repr(x) == "ReLU()":
          x.register_forward_hook(activation_hook)
      for x in i.preprocess1.op:
        if repr(x) == "ReLU()":
          x.register_forward_hook(activation_hook)
    
    #print("Activation Functions: {} {}".format(act1, act2))
    for j in i._ops:
      if "Conv" in repr(j) and "FactorizedReduce" not in repr(j):
        for j_i in j.op:
          if repr(j_i ) == "ReLU()":
            j_i.register_forward_hook(activation_hook)
  
      
  
  from sklearn.preprocessing import StandardScaler
  
  
  
  ##Generate Activation Patterns
  N = 64
  c_list = list()
  for _ in range(N):
  
    C = list()
    batch = torch.rand(1,16,8)
    model(batch)
    c_out = torch.cat(C,dim = 1 )
    c_list.append(c_out)
  K_h = hamming_kernel(c_list)
  SS = StandardScaler()
  K_h_out = SS.fit_transform(K_h.detach().numpy())
  scores.append(score(K_h))
  print("Final Score: {} -- acc : {}".format(score(K_h), a_acc))
plt.plot(scores)
plt.plot(acc)
plt.plot(rec)
plt.show()


    
      
