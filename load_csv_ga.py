
# %% 
import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
def read_data():
    scores = {}
    recall = {}
    configs = {}
    pop_size = 100
    total = 2500
    sample_size = 25
    batch = 8
    with open( "/home/snaags/RegEvo_new.csv" , newline = "") as csvfile:
        reader = csv.reader(csvfile, delimiter = ",")
        generation = 0
        scores[generation] = []
        configs[generation] = []
        recall[generation] = []

        for c ,row in enumerate(reader):
            if c >= pop_size and (c - pop_size)%batch == 0:
                generation += 1
                scores[generation] = []
                configs[generation] = []
                recall[generation] = []
            scores[generation].append(float(row[0]))
            recall[generation].append(float(row[1]))
            configs[generation].append(eval("".join(row[2:])))
    return scores, recall , configs
# %%



# %%
scores ,recall ,configs = read_data()
# %%
keys = list(configs[0][0].keys())
del keys[-1]
idx = 0 
for gen in configs:
    for config in configs[gen]:
        
        for i in keys:
            df1[i][idx] = config[i] 

        idx += 1
#%%
import matplotlib.colors as mcolors
def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    def hex_to_rgb(value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


    def rgb_to_dec(value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v/256 for v in value]




    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp
    
# %%
arr = df1.to_numpy()
tsne = TSNE()
arr_transformed = tsne.fit_transform(arr)
# %%

for i in keys:
    if "type" in i:
        df1[i] = df1[i].apply(lambda x : types_dict[x])


# %%
hex_list = ["#F20915", "#EB870E", "#EBC20E", "#EBE70E" , "#C2EB0E","#34E319"]
cmap = get_continuous_cmap(hex_list)
c = [0,0]
gens = []
for gen in scores:
    c_new = c[-1] + len(scores[gen])
    c.append(c_new)
    gens.append(scores[gen])
    a = 1
    for i in range(len(gens)):
        if i == 40:
            break

        plt.xlim(-60,60)
        plt.ylim(-60,80)
        plt.axis('off')
        plt.scatter(arr_transformed[c[-(i+2)]:c[-(i+1)],0], 
            arr_transformed[c[-(i+2)]:c[-(i+1)],1], c = gens[-(i+1)] , vmin = 0.6 ,
            vmax= 0.76, cmap = cmap, s = 5,alpha = a)
        a -=0.025
    plt.savefig("gen/{}.png".format(gen), dpi = 1500)
plt.show()




    
# %%

def plot():
    plt.scatter(arr_transformed[c[-2]:c[-1],0], 
        arr_transformed[c[-2]:c[-1],1], c = scores[gen] , vmin = 0.6 , vmax= 0.76, cmap = cmap, s = 3)

