from HPO.searchspaces.resnet import build_resnet, ResNet50
import random
#ResNet
#Cell
#Hierarchical
#Graph

def random_ops_unweighted(ops, data):
  for i in ops:
    if "OP" in i:
      ops[i] = random.choice(data["EDGE_OPERATIONS"])
  return ops

def random_activation_unweighted(ops,data):
  for i in ops:
    if "activation" in i:
      ops[i] = random.choice(data["ACTIVATION_FUNCTIONS"])
  return ops

def random_normalisation_unweighted(ops,data):
  for i in ops:
    if "normalisation" in i:
      ops[i] = random.choice(data["NORMALISATION_FUNCTIONS"])
  return ops


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


def init_config():
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
   

