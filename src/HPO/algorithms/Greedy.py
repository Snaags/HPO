from HPO.utils.visualisation import plot_scores
import csv
from ConfigSpace import ConfigurationSpace
from HPO.algorithms.algorithm_utils import train_eval,load
import json
import copy
import random
import time
def main(worker, configspace : ConfigurationSpace, json_config):
  with open(json_config) as f:
    SETTINGS = json.load(f)["SEARCH_CONFIG"]
  
  TIME_SINCE_IMPROVE = 0
  EARLY_STOP = SETTINGS["EARLY_STOP"]
  train = train_eval( worker , json_config)
  if SETTINGS["RESUME"]:
    data = load(SETTINGS["EXPERIMENT_NAME"])
    history_scores = data["scores"]
    history_conf = data["config"]
  else:
    configs = configspace.sample_configuration(SETTINGS["INITIAL_POPULATION_SIZE"])
    scores , recall , pop= train.init_async(configs)
    history_scores = scores
    history_conf = pop
    last_max_indexs = None
    iteration = 0
  while iteration < SETTINGS["TOTAL_EVALUATIONS"]:
    scores ,recall , pop = [], [], []
    while len(scores) == 0:
      time.sleep(0.5)
      scores ,recall , pop = train.get_async()
    history_conf.extend(pop)
    history_scores.extend(scores)
    print("Generation: {}".format(iteration))
    best_score = max(history_scores)
    max_indices = [index for index, value in enumerate(history_scores) if value == best_score]
    #idx_best = history_scores.index(best_score)
    print("Best Score: {}".format(best_score))
    best_configs = [history_conf[i] for i in max_indices ]
    configs = []
    while train.config_queue.qsize() < SETTINGS["CORES"]/2:
      train.update_async(configspace.mutate_graph(random.choice(best_configs),2))
    if max_indices == last_max_indexs:
      TIME_SINCE_IMPROVE+=len(scores)
    else:
      TIME_SINCE_IMPROVE = 0
    last_max_indexs = copy.copy(max_indices)
    if TIME_SINCE_IMPROVE > EARLY_STOP:
      break
    iteration+= 1

  #plot_scores(scores)
  
  best_score = max(scores)
  best_config = pop[scores.index(max(scores))]
  best_rec = recall[scores.index(max(scores))]

def main_batch(worker, configspace : ConfigurationSpace, json_config):
  with open(json_config) as f:
    SETTINGS = json.load(f)["SEARCH_CONFIG"]
  
  GENERATIONS = int(SETTINGS["TOTAL_EVALUATIONS"] / SETTINGS["CORES"])
  TIME_SINCE_IMPROVE = 0
  EARLY_STOP = 150
  train = train_eval( worker , json_config)
  configs = configspace.sample_configuration(SETTINGS["CORES"])
  scores ,recall , pop= train.eval(configs)
  history_scores = scores
  history_conf = pop
  last_max_indexs = None
  iteration = 0
  while True:
    print("Generation: {}".format(iteration))
    best_score = max(history_scores)
    max_indices = [index for index, value in enumerate(history_scores) if value == best_score]
    #idx_best = history_scores.index(best_score)
    print("Best Score: {}".format(best_score))
    best_configs = [history_conf[i] for i in max_indices ]
    configs = []
    for i in range(SETTINGS["CORES"]):
      configs.append(configspace.mutate_graph(random.choice(best_configs)))
      print("Got {} configuration".format(len(configs)))
    scores ,recall , pop= train.eval(configs)
    history_conf.extend(pop)
    history_scores.extend(scores)
    if max_indices == last_max_indexs:
      TIME_SINCE_IMPROVE+=SETTINGS["CORES"]
    else:
      TIME_SINCE_IMPROVE = 0
    last_max_indexs = copy.copy(max_indices)
    if TIME_SINCE_IMPROVE > EARLY_STOP:
      break
    iteration+= 1

  #plot_scores(scores)
  
  best_score = max(scores)
  best_config = pop[scores.index(max(scores))]
  best_rec = recall[scores.index(max(scores))]
 
