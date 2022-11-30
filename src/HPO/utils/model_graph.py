import torch.nn as nn
from HPO.utils.operations import *
import networkx as nx 
class SEMIX(nn.Module):
  def __init__(self, C_in,C_out,r =2 ,stride =1,affine = True ):
    super(SE,self).__init__()
    self.GP = nn.AdaptiveAvgPool1d(1)
    self.fc1 = nn.Linear(C_in, C_in//r, bias = False)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(C_in//r, C_out ,bias = False)
    self.sig = nn.Sigmoid()
    self.stride = stride
  def forward(self,x1,x2):
    #Squeeze
    y = self.GP(x2).squeeze()# [Batch,C]
    #torch.mean(x,axis = 2)  
    y = self.fc1(y)
    y = self.act(y)
    y = self.fc2(y)
    y = self.sig(y).unsqueeze(dim = 2)
    return x1* y.expand_as(x1)


def get_reduction(edges):
  flat = {}
  nodes = set([node for edge in edges for node in edge])
  print(nodes)
  g = nx.Graph()
  g.add_edges_from(edges)
  for i in nodes:
      flat[i] = [n for n in g.neighbors(i)]
  print(flat)
  reduction = []
  for i in flat:
      if len(flat[i]) == 2:
          if len(flat[flat[i][0]]) == 2 and not i in reduction:
              print(i,flat[i])
              reduction.append(flat[i][0])
  return reduction

class ModelGraph(nn.Module):
  def __init__(self,n_features, n_channels, n_classes, graph : list, OPS : list,binary = False):
    self.states = {}
    self.ops = nn.ModuleList()
    self.combine_ops = nn.ModuleDict() 
    self.global_pooling = nn.AdaptiveAvgPool1d(1)
    self.reduction = get_reduction(graph)
    for edge, _op in zip(graph,OPS):
      if edge[0] == "S":#INIT CHANNELS
        C = n_channels
      if edge[0] in self.reduction:
        stride = 2
        C  = c_curr
      op = OPS[name](C, stride, True)
      self.ops.append()
    if binary == True:
      self.binary = binary
      self.classifier = nn.Linear(C_prev, 1)
    else:
      self.classifier = nn.Linear(C_prev, n_classes)
  
  def _compile(self,length):
    #CHANNELS SHOULD INIT TO N_CHANNEL 
    #REDUCTION SHOULD OCCUR ALONG PATHS if a node has 1 input and 1 output and the previous node is not reduction
    batch = 32
    x = torch.rand(size = (batch, self.n_features,length))

 
  def combine(self,x1,x2):
    """
    Accepts two tensors [B,C,L] and returns [B,C,L1,L2]
    """
    batch_size  = x1.shape[0]
    channels = x1.shape[1]
    out = torch.bmm(x1.view(-1,x1.shape[-1],1),x2.view(-1,1,x2.shape[-1]))
    return out.view(batch_size,channels,out.shape[-2],out.shape[-1])


  def forward(self,x):
    self.states["S"] = x
    for iteration,(op , edge) in enumerate(zip(self.ops ,self.graph)):
      h = op(self.states[edge[0]])
      #CASE 1 - 1 INPUT
      if not (edge[1] in self.states.keys()):
        self.states[edge[1]] = h
      #CASE 2 - 2 INPUTS OF SAME SIZE (ADD)
      elif self.states[edge[1]].shape = h.shape:
        self.states[edge[1]] += h
      #CASE 3 - 2 INPUTS, SAME LENGTH (CONCAT CHANNELS)
      elif self.states[edge[1]].shape[2] = h.shape[2]:
        self.states[edge[1]] = torch.cat((self.states[edge[1]], h),dim = 1)
      #CASE 4 - 2 INPUTS SAMPLE CHANNELS (MATMUL 2D CONV)
      elif self.states[edge[1]].shape[1] = h.shape[1]:
        h = self.combine(self.states[edge[1]], h)
        if not iteration in self.combine_ops:
          kernel = h.shape[-2:]
          channels = h.shape[1]
          kernel[torch.argmax(kernel)] = 1
          self.combine_ops[iterations] = nn.conv2d(channels,channels,kernel_size,groups = channels)
        self.states[edge[1]] = self.combine_ops[iteration](self.states[edge[1]],h)
      #CASE 5 - DIFFERENT C AND L (SE NETWORK)
      else:
        if not iteration in self.combine_ops:
          self.combine_ops[iterations] = SEMIX(channels_in,channels_out)
        self.states[edge[1]] = self.combine_ops[iteration](self.states[edge[1]],h)
      
    
    
