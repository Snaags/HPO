import ConfigSpace as CS 
import ConfigSpace.hyperparameters as CSH
import os
import csv
import time 
from HPO.utils.ConfigStruct import Parameter, Cumulative_Integer_Struct, LTP_Parameter 

"""	TODO
Seperate Pooling and Convolution Layers
Add more convolution operations (kernalSize and maybe stride)
"""
 
def init_config():

  cs = CS.ConfigurationSpace()

  conv_ops= [ 
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5'
  ]
  

  ###DARTS###
  normal_node_0_0 = CSH.CategoricalHyperparameter('normal_node_0_0', choices=conv_ops)
  normal_node_0_1 = CSH.CategoricalHyperparameter('normal_node_0_1', choices=conv_ops)
  normal_index_0_0 = CSH.UniformIntegerHyperparameter(name = "normal_index_0_0", lower = 0, upper = 1)
  normal_index_0_1 = CSH.UniformIntegerHyperparameter(name = "normal_index_0_1", lower = 0, upper = 1)

  normal_node_1_0 = CSH.CategoricalHyperparameter('normal_node_1_0', choices=conv_ops)
  normal_node_1_1 = CSH.CategoricalHyperparameter('normal_node_1_1', choices=conv_ops)
  normal_index_1_0 = CSH.UniformIntegerHyperparameter(name = "normal_index_1_0", lower = 0, upper = 2)
  normal_index_1_1 = CSH.UniformIntegerHyperparameter(name = "normal_index_1_1", lower = 0, upper = 2)

  normal_node_2_0 = CSH.CategoricalHyperparameter('normal_node_2_0', choices=conv_ops)
  normal_node_2_1 = CSH.CategoricalHyperparameter('normal_node_2_1', choices=conv_ops)
  normal_index_2_0 = CSH.UniformIntegerHyperparameter(name = "normal_index_2_0", lower = 0, upper = 3)
  normal_index_2_1 = CSH.UniformIntegerHyperparameter(name = "normal_index_2_1", lower = 0, upper = 3)

  normal_node_3_0 = CSH.CategoricalHyperparameter('normal_node_3_0', choices=conv_ops)
  normal_node_3_1 = CSH.CategoricalHyperparameter('normal_node_3_1', choices=conv_ops)
  normal_index_3_0 = CSH.UniformIntegerHyperparameter(name = "normal_index_3_0", lower = 0, upper = 4)
  normal_index_3_1 = CSH.UniformIntegerHyperparameter(name = "normal_index_3_1", lower = 0, upper = 4)


  reduction_node_0_0 = CSH.CategoricalHyperparameter('reduction_node_0_0', choices=conv_ops)
  reduction_node_0_1 = CSH.CategoricalHyperparameter('reduction_node_0_1', choices=conv_ops)
  reduction_index_0_0 = CSH.UniformIntegerHyperparameter(name = "reduction_index_0_0", lower = 0, upper = 1)
  reduction_index_0_1 = CSH.UniformIntegerHyperparameter(name = "reduction_index_0_1", lower = 0, upper = 1)

  reduction_node_1_0 = CSH.CategoricalHyperparameter('reduction_node_1_0', choices=conv_ops)
  reduction_node_1_1 = CSH.CategoricalHyperparameter('reduction_node_1_1', choices=conv_ops)
  reduction_index_1_0 = CSH.UniformIntegerHyperparameter(name = "reduction_index_1_0", lower = 0, upper = 2)
  reduction_index_1_1 = CSH.UniformIntegerHyperparameter(name = "reduction_index_1_1", lower = 0, upper = 2)

  reduction_node_2_0 = CSH.CategoricalHyperparameter('reduction_node_2_0', choices=conv_ops)
  reduction_node_2_1 = CSH.CategoricalHyperparameter('reduction_node_2_1', choices=conv_ops)
  reduction_index_2_0 = CSH.UniformIntegerHyperparameter(name = "reduction_index_2_0", lower = 0, upper = 3)
  reduction_index_2_1 = CSH.UniformIntegerHyperparameter(name = "reduction_index_2_1", lower = 0, upper = 3)

  reduction_node_3_0 = CSH.CategoricalHyperparameter('reduction_node_3_0', choices=conv_ops)
  reduction_node_3_1 = CSH.CategoricalHyperparameter('reduction_node_3_1', choices=conv_ops)
  reduction_index_3_0 = CSH.UniformIntegerHyperparameter(name = "reduction_index_3_0", lower = 0, upper = 4)
  reduction_index_3_1 = CSH.UniformIntegerHyperparameter(name = "reduction_index_3_1", lower = 0, upper = 4)


  ###Optimiser###



  lr =CSH.UniformFloatHyperparameter(name = "lr",			lower = 0.000001  ,upper = 0.05)
  p =CSH.UniformFloatHyperparameter(name = "p",			lower = 0.01 ,upper = 0.3 )
  epochs = CSH.UniformIntegerHyperparameter(name = "epochs", lower = 1, upper = 5)
  layers = CSH.UniformIntegerHyperparameter(name = "layers", lower = 3, upper = 6)
  c1 = CSH.UniformFloatHyperparameter(name = "c1_weight" , lower = 1,upper = 3)
  batch_size = CSH.UniformIntegerHyperparameter(name = "batch_size", lower = 2, upper = 16)
  T_0 = CSH.UniformIntegerHyperparameter(name = "T_0", lower = 2, upper = 10)
  T_mult = CSH.UniformIntegerHyperparameter(name = "T_mult", lower = 1, upper = 3)
  channels = CSH.UniformIntegerHyperparameter(name = "channels", lower = 16, upper = 32)
  augmentations = CSH.UniformIntegerHyperparameter(name = "augmentations", lower = 0, upper = 20)
  
    ###Topology Definition]###
  
  hp_list = [
        layers,
        channels,
        normal_node_0_0 ,
        normal_node_0_1 ,
        normal_index_0_0,
        normal_index_0_1,
        normal_node_1_0 ,
        normal_node_1_1 ,
        normal_index_1_0,
        normal_index_1_1,
        normal_node_2_0 ,
        normal_node_2_1 ,
        normal_index_2_0,
        normal_index_2_1,
        normal_node_3_0 ,
        normal_node_3_1 ,
        normal_index_3_0, 
        normal_index_3_1,
        reduction_node_0_0 ,
        reduction_node_0_1 ,
        reduction_index_0_0,
        reduction_index_0_1,
        reduction_node_1_0 ,
        reduction_node_1_1 ,
        reduction_index_1_0,
        reduction_index_1_1,
        reduction_node_2_0 ,
        reduction_node_2_1 ,
        reduction_index_2_0,
        reduction_index_2_1,
        reduction_node_3_0 ,
        reduction_node_3_1 ,
        reduction_index_3_0, 
        reduction_index_3_1]
  cs.add_hyperparameters(hp_list)
  return cs

if __name__ == "__main__":
  from HPO.utils.DARTS_utils import config_space_2_DARTS
  configS = init_config()	
  print(configS.get_hyperparameters())
  c = configS.sample_configuration()
  print(c)
  print(config_space_2_DARTS(c))
