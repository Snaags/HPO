import random
import numpy as np
import networkx as nx
import copy
from enum import Enum

class Path:
  def __init__(self,nodes):
    self.nodes = nodes
    self.end_index = len(self.nodes) - 1
    self.current_index = 0
    
  def get_unknown_states(self):
    return self.nodes[self.current_index+1:]

  def _next(self):
    source = self.nodes[self.current_index]
    self.current_index +=1
    destination = self.nodes[self.current_index]
    return (source,destination)
  
  def try_next(self, blocked : list):
    if self.nodes[self.current_index] in blocked:
      return False 
    elif self.current_index < self.end_index:
      return self._next()
    else:
      return True


def topological_sort(V, E):
  """
  L ← Empty list that will contain the sorted elements
  S ← Set of all nodes with no incoming edge
  
  while S is not empty do
      remove a node n from S
      add n to L
      for each node m with an edge e from n to m do
          remove edge e from the graph
          if m has no other incoming edges then
              insert m into S
  
  if graph has edges then
      return error   (graph has at least one cycle)
  else 
      return L   (a topologically sorted order)
  """
  S = ["S"]
  L = []

  while len(S) > 0:
      n = S.pop()
      L.append(n)
      for e in g.neighbours(n):
        pass

def get_sorted_edges(edges):
  g = nx.DiGraph()
  g.add_edges_from(edges)
  sorted_edges = []
  node_generate = nx.topological_sort(g)
  node_l = [n for n in node_generate]
  node_generate = nx.topological_sort(g)
  while len(edges) > 0:
    for V in node_generate:
      edges = [ e for e in edges if e not in sorted_edges]
      for E in edges:
        if E[0] == V:
          sorted_edges.append(E)

  return sorted_edges
     

class Order: 
  """
  class for defining the order in which the edges should be computed
  """
  def __init__(self,edges):
    self.nodes = flatten(edges)
    self.node_paths = []
    g = nx.DiGraph()
    g.add_edges_from(edges)
    traverse(["S"],g, self.node_paths)
    self.paths = []
    for path in self.node_paths:
      self.paths.append(Path(path))
    self.sorted_edges = []
    self.states = []
    print(len(self.paths))
    while sum(self.states) < len(self.paths):
      self.iterate_paths()
  def get_edges(self):
    return list(dict.fromkeys(self.sorted_edges))

  def get_path_edges(self):
    out_paths = []

    for path in self.node_paths:
        hold = []
        for c,node in enumerate(path[:-1]):
            hold.append(node,path[c+1])

  def iterate_paths(self):
    blocked = []
    self.states = []
    for i in self.paths:
      blocked += i.get_unknown_states()
    for i in self.paths:
      result = i.try_next(blocked)
      if type(result) == bool:
        self.states.append(result)
      else:
        self.sorted_edges.append(result)
    return  

    """
    1-2, 2-3, 3-5, 5-6
    1-2, 2-3, 3-4, 4-5, 5-6
    a node cannot be used as an input until its use as an output in all path has been computed
    """
    


def flatten(l):
    return [item for sublist in l for item in sublist]

def get_reduction(edges,step):
    g = nx.Graph()
    g.add_edges_from(edges)
    nodes = set([node for edge in edges for node in edge])    
    neighbours = {}
    for i in nodes:
        neighbours[i] = [n for n in g.neighbors(i)]
    for i_ in neighbours:
        for i in neighbours[i_]:
            neighbours[i_][neighbours[i_].index(i)] = [n for n in g.neighbors(i)]  
    for i in neighbours:
        h = flatten(neighbours[i])
        if len(h)  == 4:
            neighbours[i] = True
        else:
            neighbours[i] = False
    return neighbours


def get_valid(node,g):

    ROUTES = []
    invalid_nodes = []
    traverse(["S"],g,ROUTES)
    for path in ROUTES:
        if node in path:
            idx = path.index(node)
            invalid_nodes.extend(path[:idx+1])

    return set(invalid_nodes)


def get_end(nodes,g):

    incomplete = []
    for node in nodes:
        ROUTES = []
        traverse([node],g)
        if len(ROUTES) == 0:
            incomplete.append(node)

    return list(set(incomplete))

def traverse(x, g, ROUTES):
    path = [n for n in g.neighbors(x[-1])]
    if len(path) > 1:
        paths = []
        for i in path:
            if i == "T":
                t = copy.copy(x)
                t.append(i)
                ROUTES.append(t)

            t = copy.copy(x)
            t.append(i)
            traverse(t,g,ROUTES)

        return paths
    elif path == ["T"]:
        x += path
        ROUTES.append(x)
    elif len(path) == 1:
        x += path
        traverse(x,g,ROUTES)

def gen_iter(edges,g = None, rate=0.2,enable_new_sources = False):
    if g == None:
      g = nx.DiGraph(edges)
    if random.random() > rate:
        #NEW PATH BETWEEN EXISTING NODES
        sorted_nodes = [x for x in nx.topological_sort(g)]
        number_valid = len(sorted_nodes) - 2 #["T"] is not valid as start and index at 0 
        start_index = random.randint(0,number_valid)
        new_source = sorted_nodes[start_index]
        existing = [n for n in g.neighbors(new_source)]
        valid = [n for n in sorted_nodes[start_index+1:] if not n in existing and not "S" in str(n)]
        if len(valid) != 0:
          new_end = random.choice(valid)
          edges.append((new_source,new_end))
        else:
          #INSERT NODE
          edge = random.choice(edges)
          NEW_ID = len(edges)+1
          idx = edges.index(edge)
          edges[idx] = (edge[0],NEW_ID)
          edges.append((NEW_ID,edge[1]))
    elif random.random() > rate or not enable_new_sources:
        #INSERT NODE
        edge = random.choice(edges)
        NEW_ID = len(edges)+1
        idx = edges.index(edge)
        edges[idx] = (edge[0],NEW_ID)
        edges.append((NEW_ID,edge[1]))
    elif enable_new_sources: #NEW SOURCE
        sorted_nodes = [x for x in nx.topological_sort(g)]
        current_source_num = 1
        for i in sorted_nodes:
          if "S" in str(i):
            current_source_num += 1
        sorted_nodes_no_source = [x for x in nx.topological_sort(g) if not "S" in str(x)]
        edges.append(("S{}".format(current_source_num),random.choice(sorted_nodes_no_source)))
    

    return list(set(edges))
"""
def gen_iter(edges,g = None, rate=0.2):
    if g == None:
      g = nx.DiGraph(edges)
    if random.random() > rate:
        #NEW PATH BETWEEN EXISTING NODES
        nodes = list(set([node for edge in edges for node in edge]))
        nodes.remove("T")
        new_source = random.choice(nodes)
        nodes = set([node for edge in edges for node in edge])
        invalid = get_valid(new_source,g)
        valid = nodes - invalid
        valid = [ i for i in valid if not (new_source, i) in edges]
        print(new_source, valid)
        if len(valid) == 0:
          #INSERT NODE
          edge = random.choice(edges)
          NEW_ID = len(edges)+1
          #print("new node: {}".format(NEW_ID))
          #edges.append((edge[0],NEW_ID))
          idx = edges.index(edge)
          edges[idx] = (edge[0],NEW_ID)
          edges.append((NEW_ID,edge[1]))
        else:
          new_dest = random.choice(list(valid))
          #print("new edge: {}".format((new_source,new_dest)))
          edges.append((new_source,new_dest))
    else:
        #INSERT NODE
        edge = random.choice(edges)
        NEW_ID = len(edges)+1
        #print("new node: {}".format(NEW_ID))
        #edges.append((edge[0],NEW_ID))
        idx = edges.index(edge)
        edges[idx] = (edge[0],NEW_ID)
        edges.append((NEW_ID,edge[1]))
    return list(set(edges))
    """        



                        
