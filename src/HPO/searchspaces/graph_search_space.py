from HPO.utils.graph_utils import gen_iter
import copy
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS
import networkx as nx

class GraphConfigSpace:
  def __init__(self,n_operations = 30):
    self.init_state = [("S",1)(1,"T")]
  def sample_configuration(self,n_samples=1):
    samples = []
    while len(samples) < n_samples:
      config = copy.copy(self.init_state)
      while len(config) < n_operations:
        config = gen_iter(config)
      samples.append(config)
    return samples
  
  



def init_config(n_ops = 30):

  cs = CS.ConfigurationSpace()

  conv_ops= [ 
    'max_pool',
    'avg_pool',
    'skip_connect',
    'point_conv' ,
    'depth_conv',
    'gelu',
    'batch_norm']
    
    #'max_pool_31x31',
    #'avg_pool_31x31',
    #'max_pool_64x64',
    #'avg_pool_64x64',
    #'depth_conv_15',
    #'depth_conv_29' ,
    #'depth_conv_61' ,
    #'depth_conv_101',
    #'depth_conv_201',
    #'SE_8',
    #'SE_16',
    #'attention_space',
    #'attention_channel']
  

  ###DARTS###
  hp_list = []
  for i in range(n_ops):  
    hp_list.append(CSH.CategoricalHyperparameter('op_{}'.format(i), choices=conv_ops))
    hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_kernel'.format(i), lower = 2 , upper = 4))#kernel
    hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_stride'.format(i), lower = 1 , upper = 2))#stride
    hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_dil'.format(i), lower = 0 , upper = 4))#dilation
  cs.add_hyperparameters(hp_list)
  return cs



class GraphConfigSpace:
  def __init__(self,n_operations = 30):
    self.g = nx.DiGraph
    self.n_operations = n_operations
    self.init_state = [("S",1),(1,"T")]
    self.ops_cs = init_config(n_operations)
  def sample_configuration(self,n_samples=1):
    samples = []
    while len(samples) < n_samples:
      graph = copy.copy(self.init_state)
      rate = 1.6
      while len(graph) < self.n_operations:
        rate  -= 0.005
        self.g = nx.DiGraph()
        self.g.add_edges_from(graph)
        graph = gen_iter(graph,self.g,rate)
      ops = self.ops_cs.sample_configuration()
      
      samples.append({"graph":graph,"ops":ops.get_dictionary()})
    return samples
  
  

