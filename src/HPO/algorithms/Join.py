from HPO.utils.visualisation import plot_scores
import csv
from ConfigSpace import ConfigurationSpace
from HPO.algorithms.algorithm_utils import train_eval
import json
import networkx as nx 
import matplotlib.pyplot as plt
import random
from HPO.searchspaces.graph_search_space import Graph

def main(worker, configspace : ConfigurationSpace, json_config):
  with open(json_config) as f:
    SETTINGS = json.load(f)["SEARCH_CONFIG"]
  scores ,recall , pop = [], [], []
  history_scores = scores
  history_conf = pop
  last_max_indexs = None
  iteration = 0
  train = train_eval( worker , json_config)
  while True:
    configs = configspace.sample_configuration(SETTINGS["CORES"])
    model_1 = configs[0]
    model_2 = configs[1]
    #model_1 = convert_model(model_1,configspace.data)
    #model_2 = convert_model(model_2,configspace.data)
    configs.insert(0,join_models(model_1,model_2))
    scores ,recall , pop= train.eval(configs)
    history_conf.extend(pop)
    history_scores.extend(scores)

    top_indices = sorted(range(len(history_scores)), key=lambda i: history_scores[i], reverse=True)[:2]
    model_1 = history_conf[top_indices[0]]
    model_2 = history_conf[top_indices[1]]
    model_1 = convert_model(model_1,configspace.data)
    model_2 = convert_model(model_2,configspace.data)
    new_config = join_models(model_1,model_2)
    scores ,recall , pop = train.eval([new_config])
    history_conf.extend(pop)
    history_scores.extend(scores)


  
  best_score = max(scores)
  best_config = pop[scores.index(max(scores))]
  best_rec = recall[scores.index(max(scores))]


def plot_graph(edges):
    g= nx.DiGraph()
    g.add_edges_from(edges)
    plt.figure(figsize = (10,5))
    for i, layer in enumerate(nx.topological_generations(g)):
        for n in layer:
            g.nodes[n]["layer"] = i
    pos = nx.multipartite_layout(g,subset_key="layer", align="vertical")
    for i in pos:
        temp= pos[i]
        temp[1] += random.uniform(-0.3,0.3)

    nx.draw(
         g, edge_color='black',pos = pos , width=3, linewidths=5,
         node_size=2000, node_color='pink', alpha=1,font_size = 20
         )

    plt.axis('off')
    plt.savefig("test") 
    plt.show()
    return g


def convert_model(m,data):
  edges = m["graph"]
  ops = m["ops"]
  g = nx.DiGraph()
  g.add_edges_from(edges)
  model = Graph(edges,g.nodes,ops,data)
  return model

def join_models(model_1,model_2):

  #GET MAX VALUE IN MODEL_1 
  max_val = model_1.max_vertex + 1
  #INCREASE ALL VALUES IN MODEL_2 BY MAX VALUE
  model_2.rename_vertices(max_val)
  #COMBINE
  m_dict_1 = model_1()
  m_dict_2 = model_2()
  print("graph 1: ",m_dict_1["graph"])
  print("graph 2: ",m_dict_2["graph"])
  m_dict_1["graph"].extend(m_dict_2["graph"])
  print("graph joined: ",m_dict_1["graph"])
  m_dict_1["ops"].update(m_dict_2["ops"])
  plot_graph( m_dict_1["graph"])
  return m_dict_1

