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



  ###Optimiser###


 
  lr =CSH.UniformFloatHyperparameter(name = "lr",			lower = 0.000001  ,upper = 0.1)
  p =CSH.UniformFloatHyperparameter(name = "p",			lower = 0.01 ,upper = 0.3 )
  epochs = CSH.UniformIntegerHyperparameter(name = "epochs", lower = 30, upper = 300)
  c1 = CSH.UniformFloatHyperparameter(name = "c1_weight" , lower = 1,upper = 5)
  T_0 = CSH.UniformIntegerHyperparameter(name = "T_0", lower = 1, upper = 10)
  T_mult = CSH.UniformIntegerHyperparameter(name = "T_mult", lower = 1, upper = 3)

  ##AugParameters 

  jitter = CSH.UniformFloatHyperparameter(name = "jitter",			lower = 0.001  ,upper = 0.5)
  scaling = CSH.UniformFloatHyperparameter(name = "scaling",			lower = 0.001  ,upper = 0.5)
  window_warp_num= CSH.UniformIntegerHyperparameter(name = "window_warp_num",			lower = 2  ,upper = 10)
  jitter_rate= CSH.UniformFloatHyperparameter(name = "jitter_rate",			lower = 0.05  ,upper = 0.9)
  scaling_rate= CSH.UniformFloatHyperparameter(name = "scaling_rate",			lower = 0.05  ,upper = 0.9)
  window_warp_rate= CSH.UniformFloatHyperparameter(name = "window_warp_rate",			lower = 0.05  ,upper = 0.9)
  
    ###Topology Definition]###
  
  hp_list = [
        c1,
        epochs,
        lr,
        p,
        T_0,
        T_mult,
        jitter,
        scaling,
        jitter_rate,
        scaling_rate,
        window_warp_rate,
        window_warp_num]
  cs.add_hyperparameters(hp_list)
  return cs

if __name__ == "__main__":
  from HPO.utils.DARTS_utils import config_space_2_DARTS
  configS = init_config()	
  print(configS.get_hyperparameters())
  c = configS.sample_configuration()
  print(c)
  print(config_space_2_DARTS(c))
