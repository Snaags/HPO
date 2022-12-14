import torch.nn as nn
import torch
import numpy as np
import HPO.utils.graph_ops as OPS1D
import HPO.utils.graph_ops2d as OPS2D
import copy
import networkx as nx 
from HPO.utils.graph_utils import get_reduction, Order, get_sorted_edges

def transform_idx(original_list,original_list_permuted,new_list):
  """
  Takes in 3 arrays of the same shape and elements, an list_array which has been sorted in some way
  to make a new original_list_permuted and applies the same transform to the new_list as has been done 
  to the original_array.
  """
  transform_idx = [original_list_permuted.index(i) for i in original_list]
  print(transform_idx)
  return list(np.asarray(new_list)[transform_idx])

class ModelGraph(nn.Module):
  def __init__(self,n_features, n_channels, n_classes,signal_length, graph : list, op_graph : list,device,binary = False,data_dim = 1):
    super(ModelGraph,self).__init__()
    #INITIALISING MODEL VARIABLES
    self.device = device
    self.data_dim = data_dim
    self.n_features = n_features
    self.graph = copy.copy(graph)
    self.edges = get_sorted_edges(graph)
    self.current_iteration = -1
    self.op_graph = op_graph
    self.n_channels = n_channels
    self.OP_NAMES = []
    self.op_keys = []

    #EXTRACT OPERATIONS ORDER BASED ON GRAPH
    for i in range(len(self.graph)):
      self.OP_NAMES.append(op_graph["op_{}".format(i)])
      self.op_keys.append(i)

    #STRUCTURES FOR HOLDING DATA STATES AND OPERATIONS
    self.states = {}
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
      self.stem = nn.Conv2d(n_features,n_channels,1,stride = STEM_STRIDE ,padding = STEM_PADDING)
    else:
      self.stem = nn.Conv1d(n_features,n_channels,1) #Will just leave this at defaults for now
    self.stem = self.stem.cuda(device)

    #DEFINE OP_MODULE BASED ON DATA_DIM
    if self.data_dim == 2:
      self.OP_MODULE = OPS2D
    else:
      self.OP_MODULE = OPS1D

    #BUILDS THE OPERATIONS ALONG EDGES BASED ON N_CHANNELS OF PREVIOUS OP
    self._compile(signal_length)
       

    #BUILD CLASSIFIER
    C = self.states["T"].shape[1]
    if data_dim == 2:
      self.global_pooling = nn.AdaptiveAvgPool2d(1)
    else:
      self.global_pooling = nn.AdaptiveAvgPool1d(1)
    if binary == True:
      self.binary = binary
      self.classifier = nn.Linear(C, 1)
    else:
      self.classifier = nn.Linear(C, n_classes)
  
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

    #REORDERS THE OPERATION LIST AND THE LIST OF OPERATION PARAMETERS TO MATCH THE TOPOLOGICALLY SORTED GRAPH
    OP_NAMES_ORDERED = transform_idx(self.graph,self.edges,self.OP_NAMES)
    OP_KEYS = transform_idx(self.graph,self.edges,self.op_keys)
    

    self.states["S"] = x
    self.required_states = {}
    for iteration,(name , edge,keys) in enumerate(zip(OP_NAMES_ORDERED ,self.edges,OP_KEYS)):
     
      #print(edge,self.combine_index,self.states[edge[0]].shape,name)
      """
      #This tracks the datastates that aren't required at each step so they can be deleted 
      #but it can only be used for inference since they are all needed to calculate the gradient.
      self.required_states[iteration] = []
      for todo in self.edges[iteration:]:
        self.required_states[iteration].append(todo[0])
      """

      #GET NUMBER OF CHANNELS FROM PREVIOUS DATA STATE
      if edge[0] == "S":#INIT CHANNELS
        C = self.n_channels
      else:
        C = self.states[edge[0]].shape[1]
      
      #DEFINE THE PARAMETERS OF THE OPERATION  
      stride = self.op_graph["op_{}_stride".format(keys)]
      kernel = self.op_graph["op_{}_kernel".format(keys)]
      dil = self.op_graph["op_{}_dil".format(keys)]

      #BUILD THE OPERATION
      if self.states[edge[0]].shape[2] > (stride*kernel):
        op = self.OP_MODULE.OPS[name](C, kernel,stride,dil , True).cuda(self.device)
      else:
        op = self.OP_MODULE.OPS[name](C, kernel,1,1 , True).cuda(self.device)
        
      #ADD OP TO THE MODULE LIST AND PASS THROUGH DATA
      self.ops.append(op)
      self._forward_build(op,edge,iteration)


  def combine(self,x1,x2):
    """
    Accepts two tensors [B,C,L] and returns [B,C,L1,L2]
    """

    batch_size  = x1.shape[0]
    channels = x1.shape[1]
    out = torch.bmm(x1.view(-1,x1.shape[-1],1),x2.view(-1,1,x2.shape[-1]))
    return out.view(batch_size,channels,out.shape[-2],out.shape[-1])

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
        kernel = torch.tensor(h.shape[-2:])
        channels = h.shape[1]
        kernel[torch.argmax(kernel)] = 1
        self.combine_ops[str(iteration)] = nn.Conv2d(channels,channels,kernel,groups = channels)
      self.states[edge[1]] = self.combine_ops[self.combine_index](h).squeeze()
      
    #CASE 5 - DIFFERENT C AND L (SE OPERATION)
    else:
      channels_in = h.shape[1]
      channels_out = self.states[edge[1]].shape[1]
      self.combine_ops[str(edge)] = (self.OP_MODULE.SEMIX(channels_in,channels_out)).cuda(self.device)
      self.states[edge[1]] = self.combine_ops[str(edge)](self.states[edge[1]],h)
      self.combine_index+=1

  def _forward(self, op,edge,iteration):
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
    else:
      #try:
      self.states[edge[1]] = self.combine_ops[str(edge)](self.states[edge[1]],h)
      #except:
      #  print(h.shape,op)
      #  exit()
      self.combine_index+=1

  def forward(self,x):
    #print("Starting Training -- shape:{}".format(x.shape))
    self.combine_index = 0
    self.states = {}
    self.states["S"] = self.stem(x)
    self.current_iteration = -1
    hold = 0
    #while self.current_iteration < len(self.edges)-1:
    for iteration, (op,edge) in enumerate(zip(self.ops,self.edges)):
      hold = self.combine_index
      """
      for i in self.states:
        if not i in self.required_states[iteration] and i != "T":
          del self.states[i]
      """  
      self._forward(op,edge,iteration)
    #FC LAYER
    x = self.global_pooling(self.states["T"])
    x = self.classifier(x.squeeze())
    return x

      
    
    
