import networkx as nx


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
