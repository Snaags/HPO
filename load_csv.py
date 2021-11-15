import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
scores = []
recall = []
configs = []

with open( "RegEvo.csv" , newline = "") as csvfile:
    reader = csv.reader(csvfile, delimiter = ",")
    for row in reader:
        scores.append(float(row[0]))
        recall.append(float(row[1]))
        configs.append(eval("".join(row[2:])))


sns.set_theme(style="darkgrid")
df = pd.DataFrame()
plot_data = pd.DataFrame()
df["NAS_acc"] = scores
df["NAS_error"] = [1-x for x in scores]
e_min = 1
best_list = []
for i in df["NAS_error"]:
  if i < e_min:
    e_min = i
  best_list.append(e_min)
df["NAS_best"] = best_list
df["time"] = [x for x in range(len(scores))]
df["NAS_recall"] = recall
df["res_acc"] = [0.614]*len(scores)
df["res_error"] = [1-x for x in df["res_acc"]]
df["fcn_acc"] = [0.651]*len(scores)
df["fcn_error"] = [1-x for x in df["fcn_acc"]]
# Load an example dataset with long-form data
plot_data["NAS"] = df["NAS_best"]
plot_data["ResNet"] = df["res_error"]
plot_data["FCN"] = df["fcn_error"]
# Plot the responses for different events and regions
p = sns.lineplot(data=plot_data)
p.set(xscale = "log",xlim = (1, 2500))
plt.show()
for i in range(10):
    idx_max = scores.index(max(scores))
    print("Rank {} score: {}".format(i,scores.pop(idx_max)))
    print("Config: ", configs.pop(idx_max))
    print("\n\n")
