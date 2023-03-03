from HPO.utils.visualisation import plot_scores
import csv
from ConfigSpace import ConfigurationSpace
from HPO.algorithms.algorithm_utils import train_eval
import json
def load(PATH):
    scores = []
    recall = []
    config = []
    params = []
    with open( "{}".format(PATH) , newline = "") as csvfile:
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


def main(worker, configspace : ConfigurationSpace, json_config):
  with open(json_config) as f:
    SETTINGS = json.load(f)["SEARCH_CONFIG"]
    
  train = train_eval( worker , json_config)
  configs = load(SETTINGS["RETRAIN_PATH"])["config"]
  scores ,recall , pop= train.eval(configs)
  #print("Best Score: ", max(scores))      
  #plot_scores(scores)
  
  best_score = max(scores)
  best_config = pop[scores.index(max(scores))]
  best_rec = recall[scores.index(max(scores))]
 

