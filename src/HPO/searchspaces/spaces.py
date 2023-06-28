from HPO.searchspaces.resnet import build_resnet, ResNet50
from HPO.searchspaces.graph_search_space import GraphConfigSpace
from HPO.searchspaces.utils import *
from HPO.searchspaces.cell import build_macro, build_macro_repeat
import random
#ResNet
#Cell
#Hierarchical
#Graph



class FixedModel:
  """
  Fixed model can be used for testing or hyperparameter search etC
  """
  def __init__(self, model):
    self.model = model
  def sample_configuration(self, num):
    return [self.model] * num


class ResNetSearchSpace:
  def __init__(self, JSON):
    self.data = JSON["ARCHITECTURE_CONFIG"]
    self.EDGE_OPERATIONS = self.data["EDGE_OPERATIONS"]


  def generate_layers(self):
    x = [1]
    for i in range(random.randint(self.data["MIN_LAYERS"]-1,self.data["MAX_LAYERS"]-1)):
      if random.random() < (1/len(x)):
        x.append(1)
      else:
        x[random.randint(0,len(x)-1)] += 1
    return x
      


  def sample_configuration(self, num):
    configs = []
    for i in range(num):
      m = build_resnet(self.generate_layers())
      m["ops"] = random_ops_unweighted(m["ops"],self.data)
      m["ops"] = random_activation_unweighted(m["ops"],self.data)
      m["ops"] = random_normalisation_unweighted(m["ops"],self.data)
      configs.append(m)
    print(configs)
    return configs


def resnet_search_space(JSON_CONFIG):
  return ResNetSearchSpace(JSON_CONFIG)

def resnet50(JSON_CONFIG):
    return FixedModel(ResNet50())
     

class FixedTopology:
  def __init__(self, JSON):
    with open(JSON) as conf:
      data = json.load(conf)
    self.model = model

  def   generate_ops(self):
    ops = self.model["ops"]
    "OP"



def generate_ops(op_set, location , name ,parameter_list = None):
  if parameter_list == None:
    parameter_list =[]
  if len(op_set) == 1:
    for i in location:
      parameter_list.append(CSH.Constant("{}_{}".format(i,name),value = op_set[0]))
  else:
    for i in location:
      parameter_list.append(CSH.CategoricalHyperparameter('{}_{}'.format(i,name), choices=op_set))
  return parameter_list

class SearchSpace:
  def __init__(self,JSON):
    #LOAD JSON DATA
    GRAPH_SIZE = None
    GRAPH_RATE = None
    EDGE_OPERATIONS = None 
    
    #GENERATE GRAPH
    self.edges_op = []
    self.nodes_op = []
    self.graph = []
    self.hyperparameters = []

    #GENERATE EDGE OPS
    self.hyperparameters = generate_ops(EDGE_OPERATIONS, self.graph, "OP",self.hyperparameters)
    
    #GENERATE NODE OPS
    self.hyperparameters = generate_ops(NODE_ACTIVATIONS, self.nodes, "activation",self.hyperparameters)
    self.hyperparameters = generate_ops(NODE_NORMALISATION, self.nodes, "normalisation",self.hyperparameters)
    
    #Here the same node can be selected twice to allow for a larger downsampling
    #THIS NEEDS TO BE CHANGED!!!!
    self.hyperparameters = generate_ops(self.nodes, downsample_quantity, "downsample",self.hyperparameters)

    #This channel variation should be for that op only allowing for bottlenecks and expansions.
    self.hyperparameters = generate_ops([0.25,0.5,1,2,4], self.nodes, "channel_ratio",self.hyperparameters)
   

def graph_config(JSON):
  print("here")
  graph = GraphConfigSpace(JSON)
  return graph

class CellSpace:
  def __init__(self,JSON):
    self.data = JSON["ARCHITECTURE_CONFIG"]
    self.n_nodes = 4
    self.cells = 5
    self.reduce = 1

  def sample_configuration(self,n):
    configs = []
    for i in range(n):
      self.g = nx.DiGraph()
      model = build_macro( n_nodes = self.n_nodes, n_cells = self.cells, reduction_freq = self.reduce)
      self.g.add_edges_from(model[0])
      ops = generate_op_names(self.g)
      ops = random_ops_unweighted(ops, self.data)
      ops = random_activation_unweighted(ops,self.data)
      ops = random_normalisation_unweighted(ops,self.data)
      #ops = random_combine_unweighted(ops,self.data)
      #ops = random_strides(ops,self.data["STRIDE_COUNT"])
      ops.update(model[1])
      del ops["T_stride"]
      del ops["T_channel_ratio"]
      del ops["S_stride"]
      del ops["S_channel_ratio"]
      configs.append({"graph": model[0], "ops": ops})
    return configs

class CellSpaceRepeat:
  def __init__(self,JSON):
    self.data = JSON["ARCHITECTURE_CONFIG"]
    self.n_nodes = 4
    self.cells = self.data["N_CELLS"]
    self.reduce = 1

  def sample_configuration(self,n):
    configs = []
    for i in range(n):
      self.g = nx.DiGraph()
      model_stride = random.choice(self.data["STRIDE_RATE"])
      model_stride_channel_ratio = random.choice(self.data["CHANNEL_DEPTH_RATE"])
      model = build_macro_repeat( data = self.data, n_nodes = self.n_nodes, n_cells = self.cells, reduction_freq = self.reduce,stride = model_stride,channel_ratio = model_stride_channel_ratio)
      self.g.add_edges_from(model[0])
      #ops = generate_op_names(self.g)
      #ops = random_ops_unweighted(ops, self.data)
      #ops = random_activation_unweighted(ops,self.data)
      #ops = random_normalisation_unweighted(ops,self.data)
      #ops = random_combine_unweighted(ops,self.data)
      #ops = random_strides(ops,self.data["STRIDE_COUNT"])



       
      ops_temp = generate_skip(self.g)
      for i in ops_temp:
          if not i in model[1]:
              model[1][i] = ops_temp[i]
      #ops.update(model[1])
      del model[1]["T_stride"]
      del model[1]["T_channel_ratio"]
      del model[1]["T_channel"]
      del model[1]["S_stride"]
      del model[1]["S_channel_ratio"]

      for i in model[1]:
        if "combine" in i:
          if model[1][i] == 1:
            model[1][i] = "ADD"
      model[1]["s_rate"] = model_stride
      model[1]["s_c_ratio"] = model_stride_channel_ratio
      configs.append({"graph": model[0], "ops": model[1]})
    return configs

def cell_config(JSON):
  return CellSpace(JSON)

def cell_repeat_config(JSON):
  return CellSpaceRepeat(JSON)

