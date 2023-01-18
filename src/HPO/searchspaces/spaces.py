#ResNet
#Cell
#Hierarchical
#Graph

    hp_list = []
    for i in range(n_ops):  
      hp_list.append(CSH.CategoricalHyperparameter('op_{}'.format(i), choices=conv_ops))
      hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_kernel'.format(i), lower = 2 , upper = 4))#kernel
      hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_stride'.format(i), lower = 1 , upper = 4))#stride
      hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_dil'.format(i), lower = 0 , upper = 4))#dilation
      hp_list.append(CSH.UniformIntegerHyperparameter('op_{}_channels'.format(i), lower = 2 , upper = 5))#channels
    cs.add_hyperparameters(hp_list)

class Topology:
  def __init__(self):

class Operations:
  def __init__(self,op_set): 
      self.hp_list = []
      self.op_set
  def generate_config_space(self,names)
      for i in names: 
      hp_list.append(CSH.CategoricalHyperparameter('op_{}'.format(i), choices=conv_ops))
    

def generate_ops(op_set, names,parameter_list = None):
  if parameter_list == None:
    parameter_list =[]
  for i in names:
      parameter_list.append(CSH.CategoricalHyperparameter('{}'.format(i), choices=op_set))
  return parameter_list

class SearchSpace:
  def __init__(self,topology : str, operations : list):
    self.edges_op = []
    self.nodes_op = []
    self.graph = []
  
  def generate_search_space(self):
      
