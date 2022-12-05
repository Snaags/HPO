from HPO.utils.graph_utils import gen_iter
import copy

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
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'max_pool_31x31',
    'avg_pool_31x31',
    'max_pool_64x64',
    'avg_pool_64x64',
    'skip_connect',
    'point_conv' ,
    'depth_conv_7',
    'depth_conv_15',
    'depth_conv_29' ,
    'depth_conv_61' ,
    'depth_conv_101',
    'depth_conv_201',
    'SE_8',
    'SE_16',
    'attention_space',
    'attention_channel']
  

  ###DARTS###
  hp_list = []
  for i in range(n_ops):  
    hp_list.append(CSH.CategoricalHyperparameter('op_{}'.format(i), choices=conv_ops))


  cs.add_hyperparameters(hp_list)
  return cs



