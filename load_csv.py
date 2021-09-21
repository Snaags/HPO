import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

scores = []
configs = []
with open( "RegEvo.csv" , newline = "") as csvfile:
    reader = csv.reader(csvfile, delimiter = ",")
    for row in reader:
        scores.append(float(row[0]))
        configs.append(eval("".join(row[1:])))

for i in range(10):
    idx_max = scores.index(max(scores))
    print("Rank {} score: {}".format(i,scores.pop(idx_max)))
    print("Config: ", configs.pop(idx_max))
    print("\n\n")
