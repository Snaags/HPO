import torch.nn as nn
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


def propagate_resolution(edges, ops):
  down_sample_nodes = []
  g = nx.DiGraph()
  g.add_edges_from(edges)
  for i in ops:
    if "stride" in i:
      if ops[i] > 1:
        down_sample_nodes.append(int(i.split("_")[0]))
  res_dict = {}
  for n in g.nodes():
      res_dict[n] = 1
  nodes = list(nx.topological_sort(g))
  for i in nodes:
      if i in down_sample_nodes:
          update_list = list(nx.bfs_tree(g, source=i).nodes())
          for _nodes in update_list:
              res_dict[_nodes] *=2
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
  def __init__(self,name,data,length,channels):
    self.name = name
    self.combine = data["{}_combine".format(name)]
    self.length = length
    self.channels = int(channels)
    self.activation = data["{}_activation".format(name)]
    self.normalisation = data["{}_normalisation".format(name)]
    
  def generate_edge(self,node_previous) -> list: 
    ops = []#nn.ModuleList()
    if self.channels != node_previous.channels:
      ops.append(OPS["resample_channels"](node_previous.channels,self.channels))
    self.stride = node_previous.length // self.length
    #print(node_previous.length,self.length)
    if self.stride != 1:
      ops.append(OPS["downsample_resolution"](self.channels,self.stride))
    if self.normalisation != "none":
      ops.append(NORMALISATION[self.normalisation](self.channels,self.length)) 
    if self.activation != "none":
      ops.append(ACTIVATION[self.activation](self.channels,self.length)) 
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
  def __init__(self,n_features, n_channels, n_classes,signal_length, graph : list, ops : list, device,binary = False,data_dim = 1,sigmoid = False):
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
    self.ops = nn.ModuleList()
    self.combine_ops = nn.ModuleDict() 
    
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
      self.stem = nn.Conv1d(n_features,n_channels,1,stride = 1) #Will just leave this at defaults for now
    self.stem = self.stem.cuda(device)

    #DEFINE OP_MODULE BASED ON DATA_DIM
    #if self.data_dim == 2:
    #  self.OP_MODULE = OPS2D
    #else:
    #  self.OP_MODULE = OPS1D
  
    #BUILD NODES
    self.resolution_dict = propagate_resolution(self.sorted_graph, self.ops_list)
    self.channel_dict = propagate_channels(self.sorted_graph, self.ops_list)
    g = nx.DiGraph()
    g.add_edges_from(self.sorted_graph)
    for i in g.nodes():
      self.nodes[i] = Node(i,self.ops_list,signal_length // self.resolution_dict[i], self.channel_dict[i]*n_channels)

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
      self.actfc = nn.Tanh()
    self.sigmoid = sigmoid
  
  def _compile(self,size):
    """
    Builds the operations along edge paths, 
    """
    #GENERATE RANDOM DATA TO PASS THROUGH THE NETWORK
    batch = 16
    self.combine_index = 0
    if self.data_dim == 2:
      x = torch.rand(size = (batch, self.n_features,size,size)).cuda(self.device)
    else:
      x = torch.rand(size = (batch, self.n_features,size)).cuda(self.device)
    x = self.stem(x)

    self.states["S"] = x
    
    #ITERATE THROUGH EDGES
    for iteration,edge in enumerate(self.sorted_graph):
      if self.DEBUG:
        print(edge,self.combine_index,self.states[edge[0]].shape,name)
 
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
      edge_container.append( OPS[curr_op](C).cuda(self.device))
      for i in node_ops:
        edge_container.append(i) 
       

      #ADD OP TO THE MODULE LIST AND PASS THROUGH DATA
      self.ops.append(edge_container)
      

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
    h = op(self.states[edge[0]])
    #CASE 1 - 1 INPUT
    if not (edge[1] in self.states.keys()):
      self.states[edge[1]] = h
    #CASE 2 - 2 INPUTS OF SAME SIZE (ADD)
    elif self.nodes[edge[1]].combine == "ADD":
      self.states[edge[1]] = self.states[edge[1]] + h
    elif self.nodes[edge[1]].combine == "CONCAT":
      self.states[edge[1]] = torch.cat((self.states[edge[1]], h),dim = 1)

  def forward(self,x):
    self.combine_index = 0
    self.states = {}
    self.states["S"] = self.stem(x)
    self.current_iteration = -1
    hold = 0
    #while self.current_iteration < len(self.edges)-1:
    for iteration, (op,edge) in enumerate(zip(self.ops,self.sorted_graph)):
      self._forward(op,edge)
    #FC LAYER
    x = self.global_pooling(self.states["T"])
    x = self.classifier(x.squeeze())
    if self.sigmoid:
      x = self.actfc(x)
    return x

      
    
    
