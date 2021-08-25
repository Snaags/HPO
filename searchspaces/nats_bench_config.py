import ConfigSpace as CS 
import ConfigSpace.hyperparameters as CSH

"""	TODO
Seperate Pooling and Convolution Layers
Add more convolution operations (kernalSize and maybe stride)
"""
def fully_connected_cell( num_nodes : int, num_operations : int , operation_list : list ): 
  cs = CS.ConfigurationSpace()

  for out_node in range(1,num_nodes):
    for in_node in range(0,out_node):
      op = CSH.CategoricalHyperparameter(name =  str(in_node)+"-"+str(out_node), choices = operation_list)
      cs.add_hyperparameter(op)
  return cs

def nats_bench_topology( V : int  = 4, L : int = 5):
  OPS = ["nor_conv_3x3", "nor_conv_1x1","none" , "avg_pool_3x3", "skip_connect"]
  return fully_connected_cell(V,L,OPS)

if __name__ == "__main__":
  cs = nats_bench_topology()
  print(cs.sample_configuration())   
