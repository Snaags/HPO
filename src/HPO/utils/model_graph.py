import torch.nn as nn
import math
import torch
import numpy as np
from HPO.utils.graph_ops import OPS, ACTIVATION , NORMALISATION
import copy
import networkx as nx 
from HPO.utils.graph_utils import get_reduction, Order, get_sorted_edges
import matplotlib.pyplot as plt

def propagate_channels(edges,ops):
  down_sample_nodes = []
  g = nx.DiGraph()
  g.add_edges_from(edges)
  for i in ops:
    if "channel" in i:
        if ops[i] != 1:
          ##print(i,ops[i])
          down_sample_nodes.append((int(i.split("_")[0])))
  res_dict = {}
  for n in g.nodes():
      res_dict[n] = 1
  nodes = list(nx.topological_sort(g))
  for i in nodes:
      if i in down_sample_nodes:
          update_list = list(nx.bfs_tree(g, source=i).nodes())
          for _nodes in update_list:
              res_dict[_nodes] *= ops["{}_channel_ratio".format(i)]
  #plt.figure(figsize = (19,12))
  #nx.draw(
  #    g, edge_color='black', width=1, linewidths=1,
  #    node_size=500, node_color='pink', alpha=0.9,
  #    labels={node: "{}({})".format(node,res_dict[node]) for node in g.nodes()}
  #    )
  #plt.axis('off')
  #plt.savefig("channels")
  return res_dict

def propagate_channels_combine(edges,ops):
  down_sample_nodes = []
  combine_nodes = []
  g = nx.DiGraph()
  g.add_edges_from(edges)
  for i in ops:
    if "channel" in i and i.split("_")[0].isdigit():
          down_sample_nodes.append((int(i.split("_")[0])))
    if "combine" in i:
          if i.split("_")[0].isdigit(): 
            combine_nodes.append((int(i.split("_")[0])))
          else:
            combine_nodes.append((i.split("_")[0]))
  res_dict = {}
  for n in g.nodes():
      res_dict[n] = 1
  nodes = list(nx.topological_sort(g))
  for i in nodes:
    if i in combine_nodes:
      if ops["{}_combine".format(i)] == "CONCAT":
          predecessor_nodes = list(g.predecessors(i))
          ##print("Combine values: {}".format(predecessor_nodes))
          if len(predecessor_nodes) > 1:
            res_dict[i] *= len([res_dict[x] for x in predecessor_nodes])
      else:
        predecessor_nodes = list(g.predecessors(i))
        if len(predecessor_nodes ) == 0:
          res_dict[i] = 1 
        else:
          res_dict[i]  = max([res_dict[x] for x in predecessor_nodes])
    if i in down_sample_nodes:
      res_dict[i] *= ops["{}_channel_ratio".format(i)]
      #    update_list = list(nx.bfs_tree(g, source=i).nodes())
      #    update_list.remove(i)
      #    #print("Down Stream Nodes: {}".format(update_list))
      #    for _nodes in update_list:

      ##print(res_dict)
  ##print("combines: {}".format(res_dict))
  ##print(ops)
  return res_dict


def propagate_cat_num(edges,ops):
  down_sample_nodes = []
  combine_nodes = []
  g = nx.DiGraph()
  g.add_edges_from(edges)
  for i in ops:
    if "channel" in i and i.split("_")[0].isdigit():
          down_sample_nodes.append((int(i.split("_")[0])))
    if "combine" in i:
          if i.split("_")[0].isdigit(): 
            combine_nodes.append((int(i.split("_")[0])))
          else:
            combine_nodes.append((i.split("_")[0]))
  res_dict = {}
  for n in g.nodes():
      res_dict[n] = 1
      ##print(n)
  nodes = list(nx.topological_sort(g))
  for i in nodes:
    if i in combine_nodes:
      ##print("Node: {}".format(i))
      if ops["{}_combine".format(i)] == "CONCAT":
          predecessor_nodes = list(g.predecessors(i))
          #print("Combine values: {}".format(predecessor_nodes))
          if len(predecessor_nodes) > 1:
            res_dict[i] = len([res_dict[x] for x in predecessor_nodes])
    
  return res_dict


def propagate_resolution(edges, ops):
  ##print(edges)
  down_sample_nodes = []
  g = nx.DiGraph()
  g.add_edges_from(edges)
  for i in ops:
    if "stride" in i:
      if ops[i] > 1:
        ##print(i,ops[i])
        down_sample_nodes.append(int(i.split("_")[0]))
  res_dict = {}
  for n in g.nodes():
      res_dict[n] = 1
  nodes = list(nx.topological_sort(g))
  for i in nodes:
      if i in down_sample_nodes:
          update_list = list(nx.bfs_tree(g, source=i).nodes())
          for _nodes in update_list:
              res_dict[_nodes] *= ops["{}_stride".format(i)]
  """
  plt.figure(figsize = (19,12))
  nx.draw(
      g, edge_color='black', width=1, linewidths=1,
      node_size=500, node_color='pink', alpha=0.9,
      labels={node: "{}({})".format(node,res_dict[node]) for node in g.nodes()}
      )

  plt.axis('off')
  plt.savefig("resolutions")
  """
  return res_dict

class Node:
  """
  Resolution should never increase here
  """
  def __init__(self,name,data,length,channels,cat_num):
    self.name = name
    self.combine = data["{}_combine".format(name)]
    self.length = length
    self.cat_num = cat_num
    self.channels = int(channels)
    self.activation = data["{}_activation".format(name)]
    self.normalisation = data["{}_normalisation".format(name)]

  def generate_edge(self,node_previous) -> list: 
    ops = []#nn.ModuleList()
    ##print("output channel sizes : {} {} {}".format(self.channels, self.cat_num,self.channels/self.cat_num))
    if self.channels/self.cat_num != node_previous.channels:
      ops.append(OPS["resample_channels"](node_previous.channels,int(self.channels/self.cat_num)))
    self.stride = 2**round(math.log(node_previous.length // self.length,2))
    #if self.stride > 513:
      #print("lengths:",node_previous.length,self.length,self.stride)
    if self.stride != 1:
      ops.append(OPS["downsample_resolution"](int(self.channels/self.cat_num),self.stride))
    if self.normalisation != "none":
      ops.append(NORMALISATION[self.normalisation](int(self.channels/self.cat_num),self.length)) 
    if self.activation != "none":
      ops.append(ACTIVATION[self.activation](int(self.channels/self.cat_num),self.length)) 
    return ops
    


def transform_idx(original_list,original_list_permuted,new_list):
  """
  Takes in 3 arrays of the same shape and elements, an list_array which has been sorted in some way
  to make a new original_list_permuted and applies the same transform to the new_list as has been done 
  to the original_array.
  """
  transform_idx = [original_list_permuted.index(i) for i in original_list]
  return list(np.asarray(new_list)[transform_idx])

class ModelGraph(nn.Module):
  def __init__(self,n_features, n_channels, n_classes,signal_length, 
    graph : list, ops : list, device,binary = False,data_dim = 1,sigmoid = False,
    dropout = 0.3,droppath = False,raw_stem = False,embedding = False):
    super(ModelGraph,self).__init__()
    #INITIALISING MODEL VARIABLES
    self.DEBUG = False
    self.device = device
    self.data_dim = data_dim
    self.n_features = n_features
    self.graph = copy.copy(graph)
    self.sorted_graph = get_sorted_edges(graph)
    self.current_iteration = -1
    self.n_channels = n_channels
    self.ops_list = ops

    #STRUCTURES FOR HOLDING DATA STATES AND OPERATIONS
    self.states = {}
    self.nodes  = {}
    if dropout and droppath:
      self.dropout = DropPath(dropout,self.device)
    elif dropout:
      self.dropout = nn.Dropout(dropout)
    else:
      self.dropout = nn.Identity()
    self.ops = nn.ModuleDict()
    self.combine_ops = nn.ModuleDict() 
    self.embedding = embedding
    #BUILD STEM
    """
    This is a way of increasing the capacity of a model but upping the resolution of the image off the bat.
    Not sure why the stride is 2 but that was just how it was implemented in the efficient-net implementation I saw
    Probably more useful for image data honestly, so will probably just move it inside the if statement for now
    """
    if data_dim == 2:
      STEM_PADDING = 32
      STEM_STRIDE = 2 
      self.stem = nn.Conv2d(n_features,n_channels,2,stride = STEM_STRIDE ,padding = STEM_PADDING)
    else:
      if self.embedding:
        self.stem = nn.Embedding(258,n_channels)
      else:
        if raw_stem == True:
          self.stem = nn.Conv1d(n_features,n_channels,1,stride = 1)
        else:
          self.stem = nn.Conv1d(n_features,n_channels,2,stride = 2) #Will just leave this at defaults for now
    self.stem = self.stem.cuda(device)

    #DEFINE OP_MODULE BASED ON DATA_DIM
    #if self.data_dim == 2:
    #  self.OP_MODULE = OPS2D
    #else:
    #  self.OP_MODULE = OPS1D
  
    #BUILD NODES
    self.resolution_dict = propagate_resolution(self.sorted_graph, self.ops_list)
    self.channel_dict = propagate_channels(self.sorted_graph, self.ops_list)
    self.cat_num_dict = propagate_cat_num(self.sorted_graph, self.ops_list)
    self.dropout_active = True
    ##print(self.channel_dict)
    ##print(self.resolution_dict)
    if not raw_stem:
      signal_length = signal_length//2
    g = nx.DiGraph()
    g.add_edges_from(self.sorted_graph)
    for i in g.nodes():
      self.nodes[i] = Node(i,self.ops_list,signal_length // self.resolution_dict[i], self.channel_dict[i]*n_channels,self.cat_num_dict[i])

    #BUILDS THE OPERATIONS ALONG EDGES BASED ON N_CHANNELS OF PREVIOUS OP
    self._compile(signal_length)
       

    #BUILD CLASSIFIER
    C = self.nodes["T"].channels
    if data_dim == 2:
      self.global_pooling = nn.AdaptiveAvgPool2d(1)
    else:
      self.global_pooling = nn.AdaptiveAvgPool1d(1)
    if binary == True:
      self.binary = binary
      self.classifier = nn.Linear(C, 1)
    else:
      self.classifier = nn.Linear(C, n_classes)
    if sigmoid:
      self.actfc = nn.Sigmoid()#nn.Softmax(dim =1)
    self.sigmoid = sigmoid
  
  def disable_dropout(self):
    self.dropout_active = False

  def _compile(self,size):
    """
    Builds the operations along edge paths, 
    """
    #GENERATE RANDOM DATA TO PASS THROUGH THE NETWORK
    batch = 16
    self.combine_index = 0
    if self.embedding:
      x = torch.zeros(size = (batch, self.n_features,size)).long().cuda(self.device)
    else:
      if self.data_dim == 2:
        x = torch.rand(size = (batch, self.n_features,size,size)).cuda(self.device)
      else:
        x = torch.rand(size = (batch, self.n_features,size)).cuda(self.device)      
    x = self.stem(x)

    self.states["S"] = x
    
    #ITERATE THROUGH EDGES
    for iteration,edge in enumerate(self.sorted_graph):
        #print(edge, self.ops_list["{}_{}_OP".format(edge[0],edge[1])])
        #print(self.channel_dict[edge[0]]*self.n_channels,self.channel_dict[edge[1]]*self.n_channels)
      
      #INITIALISE CONTAINER FOR EDGE OPERATIONS 
      edge_container = nn.Sequential()

      #GET OP AND NODE DATA
      curr_op = self.ops_list["{}_{}_OP".format(edge[0],edge[1])]
      curr_node = self.nodes[edge[1]]
      node_ops = curr_node.generate_edge(self.nodes[edge[0]])

      #GET NUMBER OF CHANNELS FROM PREVIOUS DATA STATE
      if edge[0] == "S":#INIT CHANNELS
        C = self.n_channels
      else:
        C = self.nodes[edge[0]].channels
      
      #BUILD THE OPERATION
      if curr_op[:2] == "L_":
        edge_container.append( OPS[curr_op](C,self.nodes[edge[0]].length).cuda(self.device))
      else:
        edge_container.append( OPS[curr_op](C).cuda(self.device))
      for i in node_ops:
        edge_container.append(i) 
       

      #ADD OP TO THE MODULE LIST AND PASS THROUGH DATA
      self.ops[str(edge)] = edge_container
      

  def combine(self,x1,x2):
    """
    Accepts two tensors [B,C,L] and returns [B,C,L1,L2]
    """

    batch_size  = x1.shape[0]
    channels = x1.shape[1]
    out = torch.bmm(x1.view(-1,x1.shape[-1],1),x2.view(-1,1,x2.shape[-1]))
    proper_size_out = out.view(batch_size,channels,out.shape[-2],out.shape[-1])
    return proper_size_out

  def next_op(self):
    self.current_iteration +=1
    op, edge = self.ops[self.current_iteration] , self.edges[self.current_iteration]
    return self.current_iteration, op, edge 
  
  def _forward_build(self,op,edge,iteration):
    h = op(self.states[edge[0]])
    #CASE 1 - 1 INPUT
    if not (edge[1] in self.states.keys()):
      self.states[edge[1]] = h
    #CASE 2 - 2 INPUTS OF SAME SIZE (ADD)
    elif self.states[edge[1]].shape == h.shape:
      self.states[edge[1]] = self.states[edge[1]] + h
    #CASE 3 - 2 INPUTS, SAME LENGTH (CONCAT CHANNELS)
    elif self.states[edge[1]].shape[2] == h.shape[2] and False:
      self.states[edge[1]] = torch.cat((self.states[edge[1]], h),dim = 1)
    #CASE 4 - 2 INPUTS SAME CHANNELS (MATMUL 2D CONV)
    elif self.states[edge[1]].shape[1] == h.shape[1] and False:
      h = self.combine(self.states[edge[1]], h)
      if not self.combine_index in self.combine_ops:
        #BUILD KERNEL OF SIZE L1,L2 THEN SETS LARGER DIM TO 3
        kernel = torch.tensor(h.shape[-2:])
        channels = h.shape[1]
        kernel[torch.argmax(kernel)] = 3
        self.combine_ops[str(edge)] = nn.Conv2d(channels,channels,kernel,groups = channels).cuda(self.device)
      self.states[edge[1]] = self.combine_ops[str(edge)](h).squeeze()
      self.combine_index+=1
      
    #CASE 5 - DIFFERENT C AND L (SE OPERATION)
    else:
      channels_in = h.shape[1]
      channels_out = self.states[edge[1]].shape[1]
      self.combine_ops[str(edge)] = (self.OP_MODULE.SEMIX(channels_in,channels_out)).cuda(self.device)
      self.states[edge[1]] = self.combine_ops[str(edge)](self.states[edge[1]],h)
      self.combine_index+=1

  def _forward(self, op,edge):
    ##print(edge,self.states[edge[0]].shape, op)
    ##print("Predicted channels: {} {}".format(self.channel_dict[edge[0]],self.channel_dict[edge[1]]))
    if self.dropout_active:
      h = op(self.dropout(self.states[edge[0]]))
    else:
      h = op(self.states[edge[0]])
    #CASE 1 - 1 INPUT
    if not (edge[1] in self.states.keys()):
      self.states[edge[1]] = h
    #CASE 2 - 2 INPUTS OF SAME SIZE (ADD)
    elif self.nodes[edge[1]].combine == "ADD":
      """
      if self.states[edge[1]].shape != h.shape:
        #print("PREADD SHAPE: {} - {}".format(self.states[edge[1]].shape, h.shape))
        #print("OP: {}".format(op))
        #print("PREVIOUS SHAPE: {}".format(self.states[edge[0]].shape))
      """
      self.states[edge[1]] = self.states[edge[1]] + h
    elif self.nodes[edge[1]].combine == "CONCAT":
      ##print("PRECONCAT: {} - {}".format(self.states[edge[1]].shape, h.shape))
      self.states[edge[1]] = torch.cat((self.states[edge[1]], h),dim = 1)
    elif self.nodes[edge[1]].combine == "MULT":
      self.states[edge[1]] = self.states[edge[1]] * h
    ##print(edge,self.states[edge[1]].shape)
    else:
      self.states[edge[1]] = self.states[edge[1]] + h
  def forward(self,x):
    self.combine_index = 0
    self.states = {}
    if self.embedding:
      x = x.squeeze().long()
      self.states["S"] = self.stem(x).permute(0,2,1)
    else: 
      self.states["S"] = self.stem(x)
    self.current_iteration = -1
    hold = 0
    #while self.current_iteration < len(self.edges)-1:
    for iteration, edge in enumerate(self.sorted_graph):
      op = self.ops[str(edge)]
      self._forward(op,edge)
    #FC LAYER
    x = self.global_pooling(self.states["T"])
    x = self.classifier(x.squeeze())
    if self.sigmoid:
      x = self.actfc(x)
    return x

class DropPath(nn.Module):
  def __init__(self, drop_prob,device):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob
    self.device = device
  def forward(self,x):
    if self.drop_prob > 0.:
      keep_prob = 1.-self.drop_prob
      mask = torch.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob).cuda(self.device)
      #x.div(keep_prob)
      x.mul(mask)
    return x


      
    
    
