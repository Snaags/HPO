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
    'sep_conv_9x9',
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


  ###Optimiser###
  lr =CSH.UniformFloatHyperparameter(name = "lr",			lower = 0.000001  ,upper = 0.05)
  p =CSH.UniformFloatHyperparameter(name = "p",			lower = 0.01 ,upper = 0.3 )
  epochs = CSH.UniformIntegerHyperparameter(name = "epochs", lower = 4, upper = 150)
  layers = CSH.UniformIntegerHyperparameter(name = "layers", lower = 2, upper = 5)
  T_0 = CSH.UniformIntegerHyperparameter(name = "T_0", lower = 1, upper = 20)
  T_mult = CSH.UniformIntegerHyperparameter(name = "T_mult", lower = 1, upper = 3)
  channels = CSH.UniformIntegerHyperparameter(name = "channels", lower = 10, upper = 40)
  



  ##AugParameters 

  jitter = CSH.UniformFloatHyperparameter(name = "jitter",      lower = 0.001  ,upper = 0.5)
  scaling = CSH.UniformFloatHyperparameter(name = "scaling",      lower = 0.001  ,upper = 0.5)
  window_warp_num= CSH.UniformIntegerHyperparameter(name = "window_warp_num",     lower = 2  ,upper = 10)
  crop = CSH.UniformFloatHyperparameter(name = "crop",      lower = 0.5  ,upper = 0.9)
  mix_up = CSH.UniformFloatHyperparameter(name = "mix_up",      lower = 0.01  ,upper = 0.9)
  cut_out = CSH.UniformFloatHyperparameter(name = "cut_out",      lower = 0.05  ,upper = 0.5)
  cut_mix = CSH.UniformFloatHyperparameter(name = "cut_mix",      lower = 0.05  ,upper = 0.5)

  crop_rate = CSH.NormalFloatHyperparameter(name = "crop_rate", lower = 0.0  ,mu = 0.5 , sigma = 0.5,upper = 5)
  jitter_rate= CSH.NormalFloatHyperparameter(name = "jitter_rate",lower = 0.0  ,mu = 0.5 , sigma = 0.5,upper = 5)
  scaling_rate= CSH.NormalFloatHyperparameter(name = "scaling_rate",lower = 0.0  ,mu = 0.5 , sigma = 0.5,upper = 5)
  window_warp_rate= CSH.NormalFloatHyperparameter(name = "window_warp_rate",lower = 0.0  ,mu = 0.5 , sigma = 0.5,upper = 5)
  mix_up_rate= CSH.NormalFloatHyperparameter(name = "mix_up_rate",lower = 0.0  ,mu = 0.5 , sigma = 0.5,upper = 5)
  cut_out_rate= CSH.NormalFloatHyperparameter(name = "cut_out_rate",lower = 0.0  ,mu = 0.5 , sigma = 0.5,upper = 5)
  cut_mix_rate= CSH.NormalFloatHyperparameter(name = "cut_mix_rate",lower = 0.0  ,mu = 0.5 , sigma = 0.5,upper = 5)
    ###Topology Definition]###



    ###Topology Definition]###
  
  hp_list = [
        epochs,
        lr,
        p,
        layers,
        T_0,
        T_mult,
        channels,
        normal_node_0_0 ,
        jitter ,
        scaling ,
        window_warp_num,
        crop ,
        mix_up ,
        cut_out ,
        cut_mix ,
        crop_rate ,
        jitter_rate,
        scaling_rate,
        window_warp_rate,
        mix_up_rate,
        cut_out_rate,
        cut_mix_rate,
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
        normal_index_3_1]

  cs.add_hyperparameters(hp_list)
  return cs

if __name__ == "__main__":
  from HPO.utils.DARTS_utils import config_space_2_DARTS
  configS = init_config()
  print(configS.get_hyperparameters())
  c = configS.sample_configuration()
  print(c)
  print(config_space_2_DARTS(c))
