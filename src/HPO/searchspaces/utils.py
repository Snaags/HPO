import networkx as nx
import random
EDGE_TEMPLATES = [
  "{}_{}_OP"]
NODE_TEMPLATES = [
  "{}_activation",
  "{}_normalisation",
  "{}_stride",
  "{}_channel_ratio",
  "{}_combine"]

def random_strides(ops,count,data,stride, channel_ratio):
  s = [ x for x in ops if "stride" in x]
  while count > 0:
    o = random.choice(s)
    ops[o] *= stride
    ops["{}_channel_ratio".format(o.split("_")[0])] *=channel_ratio # FIX THIS ALSO 
    count -= 1

  """
  for i in s:
    ops[i] = 2 ** (ops[i] - 1 )
    ops["{}_channel_ratio".format(i.split("_")[0])] = 2 ** (ops["{}_channel_ratio".format(i.split("_")[0])] - 1 )
    
  """
  return ops


def generate_skip(graph):
  ops = {}
  nodes = graph.nodes()
  edges = graph.edges()
  for template in NODE_TEMPLATES:
    for node in nodes:
        if "stride" in template or "channel" in template:
          ops[template.format(node)] = 1

        else:
          ops[template.format(node)] = "none"
  for template in EDGE_TEMPLATES:
    for edge in edges:
      ops[template.format(*edge)] = "skip_connect"
  return ops

def generate_op_names(graph):
  ops = {}
  nodes = graph.nodes()
  edges = graph.edges()
  for template in NODE_TEMPLATES:
    for node in nodes:
        if "stride" or "channel" in template:
          ops[template.format(node)] = 1
        else:
          ops[template.format(node)] = "none"
  for template in EDGE_TEMPLATES:
    for edge in edges:
      ops[template.format(*edge)] = "none"
  return ops

def random_ops_unweighted(ops, data):
  for i in ops:
    if "OP" in i:
      ops[i] = random.choice(data["EDGE_OPERATIONS"])
  return ops

def random_activation_unweighted(ops,data):
  for i in ops:
    if "activation" in i:
      ops[i] = random.choice(data["ACTIVATION_FUNCTIONS"])
  return ops

def random_combine_unweighted(ops,data):
  for i in ops:
    if "combine" in i:
      ops[i] = random.choice(data["COMBINE_OPERATIONS"])
  return ops

def random_normalisation_unweighted(ops,data):
  for i in ops:
    if "normalisation" in i:
      ops[i] = random.choice(data["NORMALISATION_FUNCTIONS"])
  return ops

def graph_joiner(old,new):
  #FLATTEN THE GRAPH TO GET HIGHEST NODE (USING NETWORKX)
  g = nx.DiGraph()
  g.add_edges_from(old)
  _max = 0
  for e in g.nodes():
    if type(e) == int:
      if int(e) > int(_max):
        _max = e
  #ADD MAX TO ALL VALUES IN NEW SET
  for i in new:
    if i[0] == "S":
      old.append((i[0] , i[1] + _max))
    else:
      old.append((i[0]+_max , i[1] + _max))


  return old

def op_joiner(old,new):
  _max = 0
  for n in old:
    for i in n.split("_"):
      if i.isdigit():
        if int(i) > int(_max):
          _max = i

  for n in new:
    split_key = n.split("_")
    for e,i in enumerate(split_key):
      if i.isdigit():
        split_key[e] = str(int(i) + int(_max))
    old["_".join(split_key)] = new[n]
  return old 
