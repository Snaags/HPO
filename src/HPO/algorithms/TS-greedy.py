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
from HPO.utils.ensemble import EnsembleManager
import sys

def full_eval(SETTINGS):
  accuracy = {}
  acc_best_single, recall,params = evaluate("{}/{}".format(SETTINGS["PATH"],"configuration.json"))
  accuracy["single_model"] = acc_best_single
  for i in [1,3,5,10]:
    be = EnsembleManager("{}/{}".format(SETTINGS["PATH"],"configuration.json"),SETTINGS["DEVICES"][0])
    be.get_ensemble(i)
    accuracy["ensemble_{}".format(i)] = be.evaluate(2)
    
  # convert dictionary to dataframe
  df = pd.DataFrame(accuracy, index=[0])

  # save to csv
  df.to_csv('{}/test_results.csv'.format(SETTINGS["PATH"]), index=False)

class model_ts:
  def __init__(self,acc ,recall , config,SETTINGS):
    self.SETTINGS = SETTINGS
    self.ID = config["ID"]
    self.mean_acc = acc
    self.config = config
    self.offspring = 0
    self.evals = 1
  def load_scores(self):
    path = "{}/{}/{}-bin.npy".format(self.SETTINGS["PATH"],"metrics",self.ID)
    x = np.load(path)
    self.a = sum(x) + 1
    self.b = len(x) - sum(x) + 1
    self.evals = len(x)/1500

  def sample(self):
    self.load_scores()
    return np.random.beta(self.a,self.b)

  def sample_mu(self):
    self.load_scores()
    return np.mean(np.random.beta(self.a,self.b,1000))

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
    return self.mu

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
  M_FORMAT = model
  with open(json_config) as f:
    SETTINGS = json.load(f)["SEARCH_CONFIG"]
  TIME_SINCE_IMPROVE = 0
  EARLY_STOP = SETTINGS["EARLY_STOP"]
  START_TIME = time.time()
  RUNTIME = SETTINGS["RUNTIME"]
  last_print = time.time()

  train = train_eval( worker , json_config)




  if SETTINGS["RESUME"]:
    data = load(SETTINGS["EXPERIMENT_NAME"])
    history.extend([M_FORMAT(s,r,p,SETTINGS) for s,r,p in zip( data["scores"] ,data["recall"] , data["config"])])
    history_scores = data["scores"]
    history_conf = data["config"]
  else:
    configs = configspace.sample_configuration(SETTINGS["INITIAL_POPULATION_SIZE"])
    scores , recall , pop= train.init_async(configs)
    
    history = [M_FORMAT(s,r,p,SETTINGS) for s,r,p in zip(scores ,recall , pop)]

    last_mean_best = None
    iteration = 0
  
  #BEGIN MAIN LOOP
  while iteration < SETTINGS["TOTAL_EVALUATIONS"]:
    scores ,recall , pop = [], [], [] 

    #WAIT FOR NEW RESULTS
    while len(scores) == 0:
      time.sleep(0.2)
      scores ,recall , pop = train.get_async()

    

    history.extend([M_FORMAT(s,r,p,SETTINGS) for s,r,p in zip(scores ,recall , pop)])

    print("Generation: {}".format(iteration))
    #Thompson Sampling
    max_index = np.argmax([i.sample() for i in history])
    mean_best = max([i.sample_mu() for i in history])
    print("Best (Mean) Score: {}".format(mean_best))

    while train.config_queue.qsize() < SETTINGS["CORES"]*2:
      if history[max_index].get_ratio() > 4:
        train.update_async(history[max_index].get_config())
      else:
        train.update_async(configspace.mutate_graph(history[max_index].get_config(),2))


    if time.time() > START_TIME + RUNTIME:
      print("Reached Total Alloted Time: {}".format(RUNTIME))
      break
    elif (time.time() - last_print) > 30:
      print("TIME LEFT: {} SECONDS".format((START_TIME + RUNTIME) - time.time() ))
      last_print = time.time()

    if mean_best == last_mean_best:
      TIME_SINCE_IMPROVE+=len(scores)
    else:
      TIME_SINCE_IMPROVE = 0
    last_mean_best = copy.copy(mean_best)
    if TIME_SINCE_IMPROVE > EARLY_STOP:
      break
    iteration+= 1

  train.kill_workers()
  full_eval(SETTINGS)
if __name__ == "__main__":
  with open(sys.argv[1]) as f:
    DATA = json.load(f)
    full_eval(DATA["SEARCH_CONFIG"])