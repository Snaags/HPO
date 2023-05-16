import csv

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
    print("Best Score: {}".format(max(scores)))
    return {"scores":scores,"recall":recall,"config":config,"error":error,"best":best_list ,"params": params}



def plot_graph(edges,ops):
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
        
    edge_labels = {e: ops[ "{}_{}_OP".format(e[0], e[1])] for e in g.edges()}
    res_dict =  propagate_resolution(edges, ops)
    nx.draw(
         g, edge_color='black',pos = pos , width=1, linewidths=5,
         node_size=2000, node_color='pink', alpha=0.9,font_size = 20,with_labels=True, labels={node: "{}({})".format(node,res_dict[node]) for node in g.nodes()}
         )
    
    nx.draw_networkx_edge_labels(
    g, pos,
    edge_labels=edge_labels,
    font_color='red',
    font_size = 15
)
    plt.axis('off')
    plt.savefig("test") 
    return g