from HPO.searchspaces.utils import graph_joiner, op_joiner
import networkx as nx



def build_bottleneck(stride = 1, step = False):
  graph = [(0,1),(1,2),(2,3),(0,3),(3,4)]
  bottleneck_ops = {
    "0_1_OP": "skip_connect",
    "1_2_OP": "conv_3",
    "2_3_OP": "skip_connect",
    "0_3_OP": "skip_connect",
    "3_4_OP": "skip_connect",
  }
  node_ops = {
    "1_activation" : "relu","1_normalisation": "batch_norm","1_stride": 1,"1_channel_ratio":0.25 if step == 0 else step,
    "2_activation" : "relu","2_normalisation": "batch_norm","2_stride": stride,"2_channel_ratio":1,
    "3_activation" : "none","3_normalisation": "batch_norm","3_stride": 1,"3_channel_ratio":4,
    "4_activation" : "relu","4_normalisation": "none","4_stride": 1,"4_channel_ratio":1,
    
  }
  ops = {**bottleneck_ops , **node_ops}
  return graph, ops


def make_layer(blocks, step,stride = 1 ):
  g,op = build_bottleneck(stride = stride,step = step)
  
  for i in range(blocks-1):
    g_i , op_i = build_bottleneck()
    g = graph_joiner(g,g_i)
    op = op_joiner(op,op_i)
  return g , op    
  

def build_resnet( layer_list):
    ops = {"S_0_OP": "max_pool_3","0_activation" : "relu","0_normalisation": "batch_norm","0_stride": 2,"0_channel_ratio":1,"S_activation": None, "S_normalisation": None, "S_channel_ratio": 1, "S_stride": 1, "S_combine" : None}
    graph = [("S",0)]
    
  
    graph_new, ops_new = make_layer(layer_list[0],step = 1 )
    graph = graph_joiner(graph,graph_new)
    ops = op_joiner(ops,ops_new)

    for layer in layer_list[1:]: 
        graph_new, ops_new = make_layer(layer, stride=2,step = 0.5)
        graph = graph_joiner(graph,graph_new)
        ops = op_joiner(ops,ops_new)
    """
        graph_new, ops_new = make_layer(layer_list[2], stride=2,step = 0.5)
        graph = graph_joiner(graph,graph_new)
        ops = op_joiner(ops,ops_new)

    graph_new, ops_new = make_layer(layer_list[3], stride=2,step = 0.5)
    graph = graph_joiner(graph,graph_new)
    ops = op_joiner(ops,ops_new)
    """
    g = nx.DiGraph()
    g.add_edges_from(graph)
    _max = max([ int(i) for i in g.nodes() if str(i).isdigit()])
    ops_out = {}
    for i in ops:
      ops_out[i.replace(str(_max),"T")] =  ops[i]
    for c,i in enumerate(graph):
      if int(i[1]) == int(_max):
        graph[c] = (i[0], "T")
     
    g = nx.DiGraph()
    g.add_edges_from(graph)
    for i in g.nodes():
      ops_out["{}_combine".format(i)] = "ADD"
    return {"graph":graph, "ops":ops_out}

def ResNet50():
    return build_resnet( [3,4,6,3])      

def ResNetDeep():
    return build_resnet( [2,2,1,1,1,1])      

def ResNet101():
    return build_resnet( [3,4,24,3])      
