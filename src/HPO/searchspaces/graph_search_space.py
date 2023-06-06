from HPO.utils.graph_utils import gen_iter
from HPO.searchspaces.utils import *
import copy
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS
import time
import networkx as nx

def is_fully_connected(G: nx.DiGraph) -> bool:
    # Count nodes with in-degree of 0 and out-degree of 0
    no_input_nodes = sum(1 for _, indeg in G.in_degree() if indeg == 0)
    no_output_nodes = sum(1 for _, outdeg in G.out_degree() if outdeg == 0)

    # Check if there is exactly one node with no inputs and one node with no outputs
    return (no_input_nodes == 1) and (no_output_nodes == 1)

def is_valid_network(G, input_node, output_node):
    # Check if there's only one input and one output node
    input_count = 0
    output_count = 0

    if not "S" in G.nodes or not "T" in G.nodes:
      return False
    """
    for node, degree in G.degree():
        if degree == 1:
            if G.out_degree(node) == 1:
                input_count += 1
                if input_count > 1:
                    return False
            elif G.in_degree(node) == 1:
                output_count += 1
                if output_count > 1:
                    return False
    
    if input_count != 1 or output_count != 1:
        return False
    """
    # Check if the graph is acyclic
    if not nx.is_directed_acyclic_graph(G):
        return False

    # Check if all nodes are part of a path between input and output nodes
    for n in G.nodes:
      try:
          # If there's a path from the input to output, then all nodes are connected
          _ = nx.shortest_path(G, n, output_node)
          _ = nx.shortest_path(G, input_node,n)
      except nx.NetworkXNoPath:
          return False

    return True

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
    self.n_operations = 32
    self.edge_options = self.data["N_EDGES"]
    self.init_state = [("S",1),(1,"T")]
  def sample_configuration(self,n_samples=1):
    samples = []
    while len(samples) < n_samples:
      graph = copy.copy(self.init_state)
      self.n_operations = random.choice(self.edge_options)
      while len(graph) < self.n_operations:
        rate = 0.5 
        self.g = nx.DiGraph()
        self.g.add_edges_from(graph)
        graph = gen_iter(graph,self.g,rate)
      self.g.add_edges_from(graph)
      model_stride = random.choice(self.data["STRIDE_RATE"])
      model_stride_channel_ratio = random.choice(self.data["CHANNEL_DEPTH_RATE"])
      if self.data["STRIDE_VARIABLE"]:
        stride = random.randint(1,self.data["STRIDE_COUNT"])
      else:
        stride = self.data["STRIDE_COUNT"]
      ops = generate_op_names(self.g)
      ops = random_ops_unweighted(ops, self.data)
      ops = random_activation_unweighted(ops,self.data)
      ops = random_normalisation_unweighted(ops,self.data)
      ops = random_combine_unweighted(ops,self.data)
      
      del ops["T_stride"]
      del ops["T_channel_ratio"]
      del ops["S_stride"]
      del ops["S_channel_ratio"]
      ops = random_strides(ops,stride,self.data,2,model_stride_channel_ratio)
      #ops["s_rate"] = model_stride
      #ops["s_c_ratio"] = model_stride_channel_ratio
      #ops["stem"] = random.choice(self.data["STEM_SIZE"])
      samples.append(Graph(graph, self.g.nodes,copy.copy(ops),self.data))
    return samples
  
  def mutate_graph(self,m,steps = 1):
    edges = m["graph"]
    ops = m["ops"]
    g = nx.DiGraph()
    g.add_edges_from(edges )
    model = Graph(edges,g.nodes,ops,self.data)
    result = False
    while result == False:
      result = self.check_valid(copy.deepcopy(model))
    #print("Time to find valid model: ",time.time()- s)
    if steps>0:
      return self.mutate_graph(result(),steps-1)
    return result

  def check_valid(self,model):
    #MAKE RANDOM EDIT AND GENERATE NX GRAPH
    model.get_random_edit()()
    graph = model()["graph"]
    g = nx.DiGraph()
    g.add_edges_from(graph)

    #CHECK GRAPH IS VALID
    valid = nx.is_directed_acyclic_graph(g) 

    if is_valid_network(g, "S", "T"):
      return model
    else:
      return False


from functools import wraps

def log_graph_change(func):
    @wraps(func)
    def wrapper(G, *args, **kwargs):
        edges_before = set(G.edges)
        nodes_before = set(G.nodes)
        result = func(G, *args, **kwargs)
        edges_after = set(G.edges)
        nodes_after = set(G.nodes)

        added_edges = edges_after - edges_before
        removed_edges = edges_before - edges_after
        if added_edges:
            print(f"{func.__name__} added: {added_edges}")
        if removed_edges:
            print(f"{func.__name__} removed: {removed_edges}")

        added_nodes = nodes_after - nodes_before
        removed_nodes = nodes_before - nodes_after

        if added_nodes:
            print(f"{func.__name__} added: {added_nodes}")
        if removed_nodes:
            print(f"{func.__name__} removed: {removed_nodes}")

        return result

    return wrapper




class Edge:
  def __init__(self,source,end,operation = None):
    self.source = source
    self.end = end
    self.operation = operation

  def __call__(self):
    out_dict = {}
    out_dict["{}_{}_OP".format(self.source,self.end)] = self.operation
    return out_dict

  def get_edge(self):
    return (self.source,self.end)

  def update(self,value):
    if type(self.source) == int:
      self.source += value
    if type(self.end) == int:   
      self.end += value

class Vertex:
  def __init__(self,ID,attributes : dict):
    for key, value in attributes.items():
        setattr(self, key, value)
    self.id = ID
    self.attributes = attributes

  def __call__(self):
    out_dict = {}
    for key, value in self.attributes.items():
      out_dict["{}_{}".format(self.id,key)] = value
    return out_dict

  def update(self,value):
    if type(self.id) == int:
      self.id += value


class Graph:
    def __init__(self,edges,nodes,ops,data):
        self.vertices = {}
        self.data = data
        self.edges = {}
        self.max_vertex = 1
        for e in edges:
          self.edges[e] = Edge(e[0],e[1],ops["{}_{}_OP".format(e[0],e[1])])
        for v in nodes:
          attr = {x.split("_")[-1] : ops[x] for x in ops if str(v) == x.split("_")[0] and not "OP" in x}
          self.vertices[v] = Vertex(v, attr)
          if type(v) != str:
            if v > self.max_vertex:
              self.max_vertex = v

    def __call__(self):
      ops = {k: v for d in self.vertices for k, v in self.vertices[d]().items()}
      ops.update({k: v for d in self.edges for k, v in self.edges[d]().items()})
      graph = [ self.edges[e].get_edge() for e in self.edges]
      return { "graph":graph,"ops":ops}


    def rename_vertices(self,value):
      for e in self.edges:
        self.edges[e].update(value)
      for v in self.vertices:
        self.vertices[v].update(value)


    def to_networkx(self):
        # Create a new directed graph using networkx
        G = nx.DiGraph()
        # Add nodes to the graph with their attributes
        for vertex_id, vertex in self.vertices.items():
            G.add_node(vertex_id, **vertex.attributes)
        # Add edges to the graph with their attributes
        for edge_key, edge in self.edges.items():
            G.add_edge(edge.source, edge.end, operation=edge.operation)
        return G
    def new_vertex(self):
      self.max_vertex += 1
      return self.max_vertex

    def get_total_stride(self):
      count = 1
      for v in self.vertices:
        if v != "S" or v != "T":
          if "stride" in self.vertices[v].attributes:
              count *= self.vertices[v].attributes["stride"]
      return count
    def get_random_edit(self):
      operations = [
        self.edge_split,
        self.edge_contraction,
        self.vertex_substitution,
        self.edge_insertion,
        self.edge_deletion,
        self.edge_substitution]
      return random.choice(operations)

    def get_vertex(self):
      return random.choice(list(self.vertices.keys()))

    def get_edge(self):
      e = random.choice(list(self.edges.keys()))
      return e[0], e[1]



    def get_dictionary(self):
      return self.__call__()


    def edge_split(self):
        u,v = self.get_edge()
        edge = self.edges.pop((u, v))
        new_vertex = self.new_vertex()
        self.vertices[new_vertex] = Vertex(new_vertex, self.generate_random_vertex())
        self.edges[(u, new_vertex)] = Edge(u, new_vertex, edge.operation)
        self.edges[(new_vertex, v)] = Edge(new_vertex, v, edge.operation)

    def edge_contraction(self):
        u,v = self.get_edge()
        edge_u_v = self.edges.pop((u, v))
        for (w, x) in list(self.edges.keys()):
            if x == v:
                edge_v_w = self.edges.pop((w, x))
                self.edges[(w, u)] = Edge(w, u, edge_v_w.operation)
            elif w == v:
                edge_v_w = self.edges.pop((w, x))
                self.edges[(u, x)] = Edge(u, x, edge_v_w.operation)
        #print(self.vertices.keys())
        #print(self.edges.keys())
        del self.vertices[v]

    def vertex_substitution(self):
        v = self.get_vertex()
        while v == "S" or v == "T":
          v = self.get_vertex()
        vertex = self.vertices[v]
        vertex.attributes = self.generate_random_vertex()

    def edge_insertion(self):
        u,v = self.get_vertex(), self.get_vertex()
        while (u, v) in self.edges:
          u,v = self.get_vertex(), self.get_vertex()
        self.edges[(u, v)] = Edge(u, v,  self.random_operation())

    def edge_deletion(self):
        u,v = self.get_edge()
        del self.edges[(u, v)]

    def edge_substitution(self):
        u,v = self.get_edge()
        edge = self.edges[(u, v)]
        edge.operation = self.random_operation()

    def random_operation(self):
      return random.choice(self.data["EDGE_OPERATIONS"])
    def generate_random_vertex(self):
      attr = {}
      attr["activation"]= random.choice(self.data["ACTIVATION_FUNCTIONS"])
      attr["combine"] = random.choice(self.data["COMBINE_OPERATIONS"])
      attr["normalisation"] = random.choice(self.data["NORMALISATION_FUNCTIONS"])
      #attr["channel_ratio"] = random.choice(self.data["CHANNEL_RATIOS"])
      attr["stride"] = random.choice(self.data["STRIDES"])
      if self.get_total_stride() >  self.data["STRIDE_COUNT"]:
        attr["stride"] = 1
      return attr

