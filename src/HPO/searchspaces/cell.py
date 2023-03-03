from HPO.searchspaces.utils import graph_joiner, op_joiner
import networkx as nx

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
    return graph, ops

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
        if layer % reduction_freq == 0 and layer != 0:
            ops["{}_channel_ratio".format(cell_outputs[-1]-(n_nodes+2))] = 2
            ops["{}_channel_ratio".format(cell_outputs[-1]-(n_nodes+1))] = 2
            ops["{}_stride".format(cell_outputs[-1]-(n_nodes+2))] = 2
            ops["{}_channel_ratio".format(cell_outputs[-1]-(n_nodes+1))] = 2

        print(graph)

        

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