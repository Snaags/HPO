from HPO.utils.graph_utils import gen_iter
from HPO.searchspaces.utils import *
import copy
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS
import networkx as nx


def get_ops(n_ops = 30):

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
    hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_stride'.format(i), lower = 1 , upper = 4))#stride
    hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_dil'.format(i), lower = 0 , upper = 4))#dilation
    #hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_channels'.format(i), lower = 2 , upper = 5))#channels
  cs.add_hyperparameters(hp_list)
  return cs





class GraphConfigSpace:
  def __init__(self,JSON):
    self.data = JSON["ARCHITECTURE_CONFIG"]
    self.g = nx.DiGraph
    self.n_operations = 30
    self.init_state = [("S",1),(1,"T")]
  def sample_configuration(self,n_samples=1):
    samples = []
    while len(samples) < n_samples:
      graph = copy.copy(self.init_state)
      rate = 1
      while len(graph) < self.n_operations:
        rate = 0.5 
        self.g = nx.DiGraph()
        self.g.add_edges_from(graph)
        graph = gen_iter(graph,self.g,rate)
      self.g.add_edges_from(graph)
      ops = generate_op_names(self.g)
      ops = random_ops_unweighted(ops, self.data)
      ops = random_activation_unweighted(ops,self.data)
      ops = random_normalisation_unweighted(ops,self.data)
      ops = random_combine_unweighted(ops,self.data)
      del ops["T_stride"]
      del ops["T_channel_ratio"]
      del ops["S_stride"]
      del ops["S_channel_ratio"]
      ops = random_strides(ops,self.data["STRIDE_COUNT"])
      ops["stem"] = random.choice(self.data["STEM_SIZE"])

      samples.append({"graph":graph,"ops":copy.copy(ops)})
    return samples
  
  

