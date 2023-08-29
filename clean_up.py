from HPO.utils.visualisation import plot_scores
import csv
from ConfigSpace import ConfigurationSpace
from HPO.algorithms.algorithm_utils import train_eval,load
from HPO.workers.load_eval import evaluate
import json
import copy
import random
import time
import pandas as pd
import numpy as np
from HPO.workers.ensemble import EnsembleManager
import sys
import os

def post_run_clean_up(data):
  def load_hps(PATH):
    scores = []
    recall = []
    config = []
    IDS = []
    with open( "{}".format(PATH) , newline = "") as csvfile:
            reader = csv.reader(csvfile, delimiter = ",")
            for row in reader:
                #scores.append(float(row[0]))
                recall.append(float(row[1]))
                config.append(eval("".join(row[2])))
                IDS.append(config[-1]["ID"])
    for ID in IDS:
        path = "{}/{}/{}".format("/".join(PATH.split("/")[:-1]),"metrics",ID)
        df = pd.read_csv(path)
        mu = df["accuracy"].mean()
        scores.append(mu)
    return scores, IDS, config
  #IDENTIFY WEIGHTS THAT WILL NOT BE USED
  path = "experiments/{}/{}/".format(data["EXPERIMENT_NAME"],"weights")
  weights = os.listdir(path)
  scores,model_id,config = load_hps("{}/{}/evaluations.csv".format("experiments",data["EXPERIMENT_NAME"]))
  indexed_lst = [(value, index) for index, value in enumerate(scores)]
  top_5_with_indices = sorted(indexed_lst, key=lambda x: x[0], reverse=True)[:5]
  score_mask = [index for value, index in top_5_with_indices]
  ID = np.asarray(model_id)[score_mask]
  print("Models in bottom 90%: {}".format(len(ID)))
  for i in weights:
    id_weight = int(i.split("-")[0])
    if not (id_weight in ID):
      os.remove("{}{}".format(path,i))

if __name__ == "__main__":
  with open(sys.argv[1]) as f:
    DATA = json.load(f)
    post_run_clean_up(DATA)
    