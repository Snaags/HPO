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

  conv_ops = ["StdConv", "Conv3", "Conv5","Conv7","MaxPool5","AvgPool5","MaxPool7","AvgPool7","SepConv3","SepConv5","SepConv7", "Identity"]
  
  ops_parameters = [
        Parameter("type",               "Categorical", lower_or_constant_value = conv_ops ), 
        LTP_Parameter("input_1",               "Integer", 0,10),
        LTP_Parameter("input_2",               "Integer", 0,10)]
  ops = Cumulative_Integer_Struct(cs,ops_parameters,"ops","num_ops","Integer",1,9)


  conv_parameters = [
        ops]
 
  Cumulative_Integer_Struct(cs,conv_parameters,"normal_cell", "num_conv","Integer", 1, 1).init() 

  conv_ops = ["FactorizedReduce"]
  
  ops_parameters = [
        Parameter("type",               "Categorical", lower_or_constant_value = conv_ops ), 
        LTP_Parameter("input_1",               "Integer", 0,10),
        LTP_Parameter("input_2",               "Integer", 0,10)]
  ops = Cumulative_Integer_Struct(cs,ops_parameters,"ops","num_ops","Integer",1,1)


  conv_parameters = [
        ops]
 
  Cumulative_Integer_Struct(cs,conv_parameters,"reduction_cell", "num_re","Integer", 1, 1).init() 
  """
  ops_type_list = ["StdConv"]
  
  ops_parameters = [
        Parameter("type",               "Categorical", lower_or_constant_value = ops_type_list ), 
        LTP_Parameter("input",               "Integer", 0,10)]
  ops = Cumulative_Integer_Struct(cs,ops_parameters,"ops","num_ops","Integer",1,5)


  conv_parameters = [
        Parameter("type",               "Constant", lower_or_constant_value = "Conv1D"),
        Parameter("padding",            "Constant" ,lower_or_constant_value = "same"),
        Parameter("filters",            "Constant", lower_or_constant_value =  1),
        Parameter("BatchNormalization", "Integer", 0,1),
        Parameter("kernel_size",        "Integer", 1,16),
        Parameter("kernel_regularizer", "Float", 1e-8,5e-1, log = True),
        Parameter("bias_regularizer",   "Float", 1e-8,5e-1, log = True),
        Parameter("activity_regularizer", "Float", 1e-8,5e-1, log = True),
        ops]
 
  Cumulative_Integer_Struct(cs,conv_parameters,"cell", "num_cells","Integer", 1, 5).init() 


    
  dense_parameters = [
        Parameter("type",               "Constant", "Dense"),
        Parameter("units",              "Integer", 1,128),
        Parameter("kernel_regularizer", "Float", 1e-8,5e-1, log = True),
        Parameter("bias_regularizer",   "Float", 1e-8,5e-1, log = True),
     self.train_dataset = Train_BTC()
       Parameter("activity_regularizer", "Float", 1e-8,5e-1, log = True)]
     
  Cumulative_Integer_Struct(cs,dense_parameters,"dense","num_dense_layers","Integer", 1, 3).init() 
  """
    ###Training Configuration###
    ###Optimiser###
  lr =CSH.UniformFloatHyperparameter(name = "lr",			lower = 0.000001  ,upper = 0.05)
  p =CSH.UniformFloatHyperparameter(name = "p",			lower = 0.01 ,upper = 0.3 )
  epochs = CSH.UniformIntegerHyperparameter(name = "epochs", lower = 1, upper = 50)
  layers = CSH.UniformIntegerHyperparameter(name = "layers", lower = 1, upper = 6)
  c1 = CSH.UniformFloatHyperparameter(name = "c1_weight" , lower = 1,upper = 3)
  batch_size = CSH.UniformIntegerHyperparameter(name = "batch_size", lower = 2, upper = 16)
  T_0 = CSH.UniformIntegerHyperparameter(name = "T_0", lower = 2, upper = 10)
  T_mult = CSH.UniformIntegerHyperparameter(name = "T_mult", lower = 1, upper = 5)
  channels = CSH.UniformIntegerHyperparameter(name = "channels", lower = 2, upper = 30)
  augmentations = CSH.UniformIntegerHyperparameter(name = "augmentations", lower = 0, upper = 20)
  
    ###Topology Definition]###
  
  hp_list = [
        c1,
        epochs,
        lr,
        p,
        layers,
        T_0,
        T_mult,
        batch_size,
        channels,
        augmentations
 
]
  cs.add_hyperparameters(hp_list)
  return cs

if __name__ == "__main__":
	configS = init_config()	
	print(configS.get_hyperparameters())
