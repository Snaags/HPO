from HPO.utils.visualisation import plot_scores
import csv
from ConfigSpace import ConfigurationSpace
from HPO.algorithms.algorithm_utils import train_eval,load
import json
import copy
import random
import time
import pandas as pd
import numpy as np




class model-ts:
  def __init__(self,acc ,recall , config,SETTINGS):
    self.SETTINGS = SETTINGS
    self.ID = config["ID"]
    self.mean_acc = acc
    self.config = config
    self.offspring = 0
    self.evals = 1
  def load_scores(self):
    path = "{}/{}/{}".format(self.SETTINGS["PATH"],"metrics",self.ID)
    try:
      x = np.load(path)
      a = sum(x) + 1
      b = len(x) - sum(x) + 1
      A = np.random.beta(a,b,1000000)
      self.evals = len(self.df["accuracy"])
    except:
      self.mu = self.mean_acc

  def sample(self):
    self.load_scores()
    return np.random.normal(self.mu,self.sigma)

  def sample_mu(self):
    self.load_scores()
    return self.mu

  def get_config(self):
    self.offspring += 1 
    return self.config

  def get_ratio(self):
    return self.offspring / self.evals


class model:
  def __init__(self,acc ,recall , config,SETTINGS):
    self.SETTINGS = SETTINGS
    self.ID = config["ID"]
    self.mean_acc = acc
    self.config = config
    self.offspring = 0
    self.evals = 1
  def load_scores(self):
    path = "{}/{}/{}".format(self.SETTINGS["PATH"],"metrics",self.ID)
    try:
      self.df = pd.read_csv(path)
      self.mu = self.df["accuracy"].mean()
      self.sigma = self.df["accuracy"].std()
      self.evals = len(self.df["accuracy"])
    except:
      self.mu = self.mean_acc

  def sample(self):
    self.load_scores()
    return np.random.normal(self.mu,self.sigma)

  def sample_mu(self):
    self.load_scores()
    return self.mu

  def get_config(self):
    self.offspring += 1 
    return self.config

  def get_ratio(self):
    return self.offspring / self.evals

def main(worker, configspace : ConfigurationSpace, json_config):
  #INITIALISATION
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
    
    history = [model(s,r,p,SETTINGS) for s,r,p in zip(scores ,recall , pop)]

    last_mean_best = None
    iteration = 0
  
  #BEGIN MAIN LOOP
  while iteration < SETTINGS["TOTAL_EVALUATIONS"]:
    scores ,recall , pop = [], [], [] 

    #WAIT FOR NEW RESULTS
    while len(scores) == 0:
      time.sleep(0.5)
      scores ,recall , pop = train.get_async()

    

    history.extend([model(s,r,p,SETTINGS) for s,r,p in zip(scores ,recall , pop)])

    print("Generation: {}".format(iteration))
    #Thompson Sampling
    max_index = np.argmax([i.sample_mu() for i in history])
    mean_best = max([i.mu for i in history])
    print("Best (Mean) Score: {}".format(mean_best))

    while train.config_queue.qsize() < SETTINGS["CORES"]/2:
      if history[max_index].get_ratio() > 4:
        train.update_async(history[max_index].get_config())
      else:
        train.update_async(configspace.mutate_graph(history[max_index].get_config(),2))
    if mean_best == last_mean_best:
      TIME_SINCE_IMPROVE+=len(scores)
    else:
      TIME_SINCE_IMPROVE = 0
    last_mean_best = copy.copy(mean_best)
    if TIME_SINCE_IMPROVE > EARLY_STOP:
      break
    iteration+= 1

  #plot_scores(scores)
  
  best_score = max(scores)
  best_config = pop[scores.index(max(scores))]
  best_rec = recall[scores.index(max(scores))]

