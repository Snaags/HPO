from HPO.searchspaces.utils import *
import networkx as nx
import random

def build_cell_graph(n_nodes = 4):
    graph = []
    ops = dict()
    source_list = [1,2]
    for end_node in range(3,n_nodes+3):
        for source_node in source_list:
            graph.append((source_node,end_node))
            ops["{}_combine".format(end_node)] = "ADD"  
        source_list.append(end_node)
        graph.append((end_node,n_nodes +3))
    ops["{}_combine".format(n_nodes +3 )] = "CONCAT" 
    graph = prune_cell(graph,n_nodes)
    graph = fix_concat_inputs(graph,n_nodes)
    return graph, ops

def fix_concat_inputs(graph,n_nodes):
    nodes = [i for i in range(3,n_nodes+3)]
    output = max(nodes) + 1 
    for edge in graph:
        if edge[0] in nodes and edge[1] in nodes and (edge[0],output) in graph:

            graph.remove((edge[0],output))

    print(graph)
    return graph


def generate_cell_ops(graph,ops,data,n_nodes = 4):
    nodes = [i for i in range(3,n_nodes+3)]
    op_edges = []
    skip_edges = []
    for edge in graph:
        if edge[1] in nodes:
            op_edges.append(edge)
        else:
            skip_edges.append(edge)

    g = nx.DiGraph()
    g.add_edges_from(op_edges)
    ops_temp = generate_op_names(g)
    ops_temp = random_ops_unweighted(ops_temp, data)
    ops_temp = random_activation_unweighted(ops_temp,data)
    ops_temp = random_normalisation_unweighted(ops_temp,data)
    for i in ops_temp:
        if not i in ops:
            ops[i] = ops_temp[i]
    g = nx.DiGraph()
    g.add_edges_from(skip_edges)
    ops_temp = generate_skip(g)
    for i in ops_temp:
        if not i in ops:
            ops[i] = ops_temp[i]

    return ops


def prune_cell(graph, n_nodes = 4):
    nodes = [i for i in range(3,n_nodes+2)]
    count_list = {}
    for i in nodes:
        count_list[i] = 0
    for edge in graph:
        if edge[1] in nodes:
            count_list[edge[1]] += 1
    while max(count_list.values()) > 2:
        edge = random.choice(graph)
        if edge[1] in nodes and count_list[edge[1]] > 2:
            graph.remove(edge)
            count_list[edge[1]] -= 1
    return graph




def replace_cell_inputs(graph,ops,outputs):
    g = nx.DiGraph()
    g.add_edges_from(graph)
    graph = nx.relabel_nodes(g, {"K-1": outputs[-1],"K-2": outputs[-2]}).edges
    for i in ops:
        if "K-1" in i:
            ops[i.replace("K-1",str(outputs[-1]))] = ops.pop(i)
        if "K-2" in i:
            ops[i.replace("K-2",str(outputs[-2]))] = ops.pop(i)
    return list(graph), ops


def build_macro( n_nodes = 8, n_cells = 5, reduction_freq = 3):
    """
    Builds the graph of a cell style search space
    """
    graph = []
    ops = {}
    cell_outputs = ["S","S"]
    for layer in range(n_cells):
        graph_new , ops_new = build_cell_graph(n_nodes)


        
        graph  = graph_joiner(graph,graph_new)
        ops  = op_joiner(ops,ops_new)
        #graph , ops = replace_cell_inputs(graph,ops,cell_outputs) #REPLACE "K-1","K-2" PLACEHOLDERS
        cell_outputs.append((n_nodes+3) + (n_nodes+3)*layer)
        graph.extend([(cell_outputs[-3],cell_outputs[-1]-((n_nodes+2))),((cell_outputs[-2],cell_outputs[-1]-(n_nodes+1)))])
        if layer % reduction_freq == 0 and layer != 0 and layer != n_cells -1 :
            ops["{}_channel_ratio".format(cell_outputs[-1]-(n_nodes+2))] = 2
            ops["{}_channel_ratio".format(cell_outputs[-1]-(n_nodes+1))] = 2
            ops["{}_stride".format(cell_outputs[-1]-(n_nodes+2))] = 2
            ops["{}_channel_ratio".format(cell_outputs[-1]-(n_nodes+1))] = 2


    g = nx.DiGraph()
    g.add_edges_from(graph)
    nodes = list(nx.topological_sort(g))
    for idx ,(e_0,e_1) in enumerate(graph):
        if e_1 == nodes[-1]:
            hold = (list(graph[idx])[0], "T")
            graph[idx] = tuple(hold)

    hold_dict = {}
    rm_list = []
    for idx ,op in enumerate(ops):
        if op.split("_")[0] == nodes[0]:
            ops[("{}_{}".format("S",op.split("_")[1]))] = ops[op]
        if int(op.split("_")[0]) == nodes[-1]:
            hold_dict[("{}_{}".format("T",op.split("_")[1]))] = ops[op]
            rm_list.append(op)
    for i in rm_list:
        del ops[i]
    ops.update(hold_dict)
    return graph, ops

def build_macro_repeat(data, n_nodes = 8, n_cells = 5, reduction_freq = 3,stride= 2, channel_ratio =2 ):
    """
    Builds the graph of a cell style search space
    """
    graph = []
    ops = {}
    cell_outputs = ["S","S"]
    graph_new , ops_new = build_cell_graph(n_nodes)
    ops_new = generate_cell_ops(graph_new,ops_new,data,n_nodes = n_nodes)
    for layer in range(n_cells):



        
        graph  = graph_joiner(graph,graph_new)
        ops  = op_joiner(ops,ops_new)
        #graph , ops = replace_cell_inputs(graph,ops,cell_outputs) #REPLACE "K-1","K-2" PLACEHOLDERS
        cell_outputs.append((n_nodes+3) + (n_nodes+3)*layer)
        graph.extend([(cell_outputs[-3],cell_outputs[-1]-((n_nodes+2))),((cell_outputs[-2],cell_outputs[-1]-(n_nodes+1)))])
        if layer % reduction_freq == 0 and layer != 0 and layer != n_cells -1 :
            ops["{}_channel_ratio".format(cell_outputs[-1]-(n_nodes+2))] = channel_ratio
            ops["{}_channel_ratio".format(cell_outputs[-1]-(n_nodes+1))] = channel_ratio
            ops["{}_stride".format(cell_outputs[-1]-(n_nodes+2))] = stride
            ops["{}_channel_ratio".format(cell_outputs[-1]-(n_nodes+1))] = channel_ratio


    g = nx.DiGraph()
    g.add_edges_from(graph)
    nodes = list(nx.topological_sort(g))
    for idx ,(e_0,e_1) in enumerate(graph):
        if e_1 == nodes[-1]:
            hold = (list(graph[idx])[0], "T")
            graph[idx] = tuple(hold)

    hold_dict = {}
    rm_list = []
    for idx ,op in enumerate(ops):
        if op.split("_")[0] == nodes[0]:
            ops[("{}_{}".format("S",op.split("_")[1]))] = ops[op]
        if int(op.split("_")[0]) == nodes[-1]:
            hold_dict[("{}_{}".format("T",op.split("_")[1]))] = ops[op]
            rm_list.append(op)
    for i in rm_list:
        del ops[i]
    ops.update(hold_dict)
    return graph, ops
            

        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    cells = 3 
    c = build_macro()
    print(c[1])
    def plot_graph(edges):
        g= nx.DiGraph()
        g.add_edges_from(edges)
        plt.figure(figsize = (50,10))
        for i, layer in enumerate(nx.topological_generations(g)):
            for n in layer:
                g.nodes[n]["layer"] = i
        pos = nx.multipartite_layout(g,subset_key="layer", align="vertical")
        for i in pos:
            temp= pos[i]
            temp[1] += random.uniform(-0.3,0.3)
    
        nx.draw(
             g, edge_color='black',pos = pos , width=1, linewidths=5,
             node_size=2000, node_color='pink', alpha=0.9,font_size = 35,with_labels=True
             )
        plt.axis('off')
        plt.savefig("test") 
    plot_graph(c[0])
    print(c)