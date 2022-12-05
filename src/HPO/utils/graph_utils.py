import random
import networkx
import copy
from enum import Enum

class Path:
  def __init__(self,nodes):
    self.nodes = nodes
    self.end_index = len(self.nodes) - 1
    self.current_index = 0
    
  def get_unknown_states(self):
    return self.nodes[self.current_index:]

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
    while sum(self.states) < len(self.paths):
      self.iterate_paths()
  def get_edges(self):
    return self.sorted_edges 
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

def get_reduction(edges):
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

def gen_iter(edges,g):
    if random.random() > 0.3:
        #INSERT NODE
        edge = random.choice(edges)
        NEW_ID = len(edges)+1
        edges.append((edge[0],NEW_ID))
        edges.append((NEW_ID,edge[1]))
        edges.remove(edge)
    else:
        #NEW PATH BETWEEN EXISTING NODES
        nodes = list(set([node for edge in edges for node in edge]))
        nodes.remove("T")
        new_source = random.choice(nodes)
        nodes = set([node for edge in edges for node in edge])
        invalid = get_valid(new_source,g)
        valid = nodes - invalid
        new_dest = random.choice(list(valid))
        edges.append((new_source,new_dest))
    return list(set(edges))
            



                        
