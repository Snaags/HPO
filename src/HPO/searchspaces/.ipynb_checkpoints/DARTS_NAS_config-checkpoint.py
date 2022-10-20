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
    'max_pool_31x31',
    'avg_pool_31x31',
    'skip_connect',
    'point_conv' ,
    'depth_conv_7',
    'depth_conv_15',
    'depth_conv_29' ,
    'depth_conv_61' ,
    'depth_conv_101',
    'depth_conv_201']
    #'patch_29' ,
    #'patch_61']
  

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

  layers = CSH.UniformIntegerHyperparameter(name = "layers", lower = 3, upper = 9)
  cut_mix_rate= CSH.UniformFloatHyperparameter(name = "cut_mix_rate",lower = 0.0  ,upper = 2)


    ###Topology Definition]###



    ###Topology Definition]###
  
  hp_list = [
        layers,
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
