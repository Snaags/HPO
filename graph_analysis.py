import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import networkx as nx
import random
import itertools
import igraph as ig
from igraph import Graph
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict


def load(FILENAME):
    scores = []
    recall = []
    config = []
    params = []
    with open( "{}".format(FILENAME) , newline = "") as csvfile:
        reader = csv.reader(csvfile, delimiter = ",")
        for row in reader:
            scores.append(float(row[0]))
            recall.append(float(row[1]))
            config.append(eval("".join(row[2])))
            if len(row) == 4:
               params.append(int(row[3])) 
    error = [1-x for x in scores]
    e_min = 1
    best_list = []
    for i in error:
      if i < e_min:
        e_min = i
      best_list.append(e_min)
    return {"scores":scores,"recall":recall,"config":config,"error":error,"best":best_list ,"params": params}


def get_formated_percentile_graphs(experiment,percentile):

    ss_1 = load("/home/cmackinnon/scripts/HPO/experiments/{}/evaluations.csv".format(experiment))

    float_list= ss_1["scores"]


    # Calculate the value of the X percentile
    percentile_value = np.percentile(float_list, 100 - percentile)

    # Get the indices of the top X percentile values
    top_percentile_indices = [i for i, value in enumerate(float_list) if value >= percentile_value]

    print("Top", percentile, "percentile indices:", top_percentile_indices)


    models = []
    for conf in np.asarray(ss_1["config"])[top_percentile_indices]:
        ops = conf["ops"]
        models.append(nx.DiGraph())
        node_vals = []
        edge_vals = []
        for i in ops:
            val = i.split("_")
            if val[0].isdigit() and not val[1].isdigit():
                node_vals.append((int(val[0]),{val[1] : ops[i] }))
            elif (val[0] == "S" or val[0] == "T") and not val[1].isdigit():
                node_vals.append((str(val[0]),{val[1] : ops[i] }))
            else:
                if (val[0] == "S") or (val[0] == "T"):
                    e0 = str(val[0])
                else:
                    e0 = int(val[0])
                if (val[1] == "S") or (val[1] == "T"):
                    e1 = str(val[1])
                else:            
                    e1 = int(val[1])


                edge_vals.append( ( e0,e1, {"OP": ops[i] })) 
        models[-1].add_edges_from(edge_vals)
        models[-1].add_nodes_from(node_vals)
    return models


def get_subgraphs(graph, subgraph_size):
    subgraphs = []
    for subgraph_vertices in itertools.combinations(graph.vs, subgraph_size):
        subgraph = graph.subgraph(subgraph_vertices)
        subgraphs.append(subgraph)
    return subgraphs


def compare_subgraphs_non_directed(graph1, graph2):
    isomorphic = graph1.isomorphic_vf2(graph2)
    return isomorphic


def compare_subgraphs_broken(graph1, graph2):
    isomorphic = graph1.isomorphic_vf2(graph2, return_mapping_12=True)
    if not isomorphic:
        return False

    # Check edge directions after finding an isomorphism
    mapping = graph1.get_isomorphism_vf2(graph2)
    for e1 in graph1.es:
        e2 = graph2.es[mapping[e1.index]]
        if e1.source != mapping[e2.source] or e1.target != mapping[e2.target]:
            return False

    return True

def compare_subgraphs(graph1, graph2):
    def edge_compat_fn(g1, g2, e1, e2):
        e1_src, e1_tgt = g1.es[e1].tuple
        e2_src, e2_tgt = g2.es[e2].tuple
        return e1_src < e1_tgt == e2_src < e2_tgt

    isomorphic = graph1.isomorphic_vf2(graph2, edge_compat_fn=edge_compat_fn)
    return isomorphic



def count_subgraph_matches(subgraph, other_graphs, subgraph_size):
    count = 0
    for other_graph in other_graphs:
        if any(compare_subgraphs(subgraph, sg) for sg in get_subgraphs(other_graph, subgraph_size)):
            count += 1
    return count


def process_graph_old(args):
    i, graph, ig_graphs, min_support, subgraph_size = args
    subgraphs = get_subgraphs(graph, subgraph_size)
    frequent_subgraphs = []
    for subgraph in subgraphs:
        count = count_subgraph_matches(subgraph, ig_graphs[i+1:], subgraph_size)
        if count >= min_support:
            frequent_subgraphs.append(subgraph)
    return frequent_subgraphs

def process_graph(args):
    i, graph, ig_graphs, min_support, subgraph_size = args
    subgraphs = get_subgraphs(graph, subgraph_size)
    frequent_subgraphs = []
    for subgraph in subgraphs:
        count = count_subgraph_matches(subgraph, ig_graphs[i+1:], subgraph_size)
        if count >= min_support:
            frequent_subgraphs.append((subgraph, count + 1))  # Add 1 to include the subgraph in the current graph
    return frequent_subgraphs

def networkx_to_igraph(nx_graph):
    # Convert node attributes to string format
    for n, attrs in nx_graph.nodes(data=True):
        for k, v in attrs.items():
            nx_graph.nodes[n][k] = str(v)
    
    # Convert edge attributes to string format
    for n1, n2, attrs in nx_graph.edges(data=True):
        for k, v in attrs.items():
            nx_graph.edges[n1, n2][k] = str(v)
    
    # Convert NetworkX graph to igraph graph
    ig_graph = ig.Graph.from_networkx(nx_graph)
    return ig_graph

# Define a function to check if an igraph graph is connected
def is_igraph_connected(graph):
    return graph.is_connected()

if __name__ == "__main__":

    experiment = "2023-03-1516:32:57.847388"
    networkx_graphs = get_formated_percentile_graphs(experiment, percentile = 10)
    ig_graphs = [networkx_to_igraph(g) for g in networkx_graphs]
    min_support = 2
    subgraph_size = 5 # Adjust this based on the desired size of subgraphs

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Prepare the list of arguments for each task
        args_list = [(i, graph, ig_graphs, min_support, subgraph_size) for i, graph in enumerate(ig_graphs)]
        
        # Execute tasks in parallel and collect results
        results = list(executor.map(process_graph, args_list))

    # Flatten the list of lists to obtain frequent subgraphs
    frequent_subgraphs = [subgraph for subgraph_list in results for subgraph in subgraph_list]
    np.save("frequent_subgraphs", np.asarray(frequent_subgraphs))
    print(f"Total frequent subgraphs found: {len(frequent_subgraphs)}")



    # Initialize a list to store unique subgraphs and their counts
    unique_subgraphs = []

    # Iterate over the frequent subgraphs and their counts
    for subgraph, count in frequent_subgraphs:
        # Check if the current subgraph is isomorphic to any of the unique subgraphs
        for i, (unique_subgraph, _) in enumerate(unique_subgraphs):
            if compare_subgraphs(subgraph, unique_subgraph):
                # Update the count of the isomorphic unique subgraph
                unique_subgraphs[i] = (unique_subgraph, unique_subgraphs[i][1] + count)
                break
        else:
            # If the subgraph is not isomorphic to any unique subgraphs, add it as a new unique subgraph
            unique_subgraphs.append((subgraph, count))

    print(f"Total unique frequent subgraphs found: {len(unique_subgraphs)}")
    np.save("unique_subgraphs", np.asarray(unique_subgraphs))


    # Filter out disconnected subgraphs
    connected_unique_subgraphs = [(subgraph, count) for subgraph, count in unique_subgraphs if is_igraph_connected(subgraph)]

    # Print the connected unique subgraphs and their counts
    print(f"Total connected unique frequent subgraphs found: {len(connected_unique_subgraphs)}")
    np.save("connected_unique_subgraphs", np.asarray(connected_unique_subgraphs))
    for subgraph, count in connected_unique_subgraphs[:30]:
        print(f"Subgraph: {subgraph}, Count: {count}")


