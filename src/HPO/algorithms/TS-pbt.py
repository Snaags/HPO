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



def clean_up_weights_end(data,active, models):
  #IDENTIFY WEIGHTS THAT WILL NOT BE USED

  results = load("{}/{}/evaluations.csv".format("experiments",data["EXPERIMENT_NAME"]))
  scores = np.asarray(results["accuracy"])
  path = "experiments/{}/{}/".format(data["EXPERIMENT_NAME"],"weights")
  weights = os.listdir(path)
  model_id = {}
  for i in models:
    model_id[i.ID] = i
  ID = np.asarray(results["ID"])
  scores = [model_id[i].sample() for i in ID]
  indexed_lst = [(value, index) for index, value in enumerate(scores)]
  top_5_with_indices = sorted(indexed_lst, key=lambda x: x[0], reverse=True)[:5]
  score_mask = [index for value, index in top_5_with_indices]
  ID = ID[score_mask]

  print("Models in bottom 90%: {}".format(len(ID)))
  for i in weights:
    id_weight = int(i.split("-")[0])
    if not (id_weight in ID):
      os.remove("{}{}".format(path,i))

def clean_unused_weights(data,active, models):
  #IDENTIFY WEIGHTS THAT WILL NOT BE USED
  results = load("{}/{}/evaluations.csv".format("experiments",data["EXPERIMENT_NAME"]))
  scores = np.asarray(results["accuracy"])
  if len(scores) > 20:
    path = "experiments/{}/{}/".format(data["EXPERIMENT_NAME"],"weights")
    weights = os.listdir(path)
    if len(weights) > 100:
      model_id = {}
      for i in models:
        model_id[i.ID] = i
      ID = np.asarray(results["ID"])
      scores = [model_id[i].sample_robust() for i in ID]
      safe_list = [model_id[i].resampled_list() for i in ID]
      score_mask = scores < np.nanquantile(scores,q = 0.90)
      resampled = ID[safe_list]
      ID = ID[score_mask]

      print("Models save via resample: {} ({})".format(len(resampled),sum(safe_list)))
      print("Models in bottom 90%: {}".format(len(ID)))
      for i in weights:
        id_weight = int(i.split("-")[0])
        if id_weight in ID and not (id_weight in active) and (not id_weight in resampled):
          os.remove("{}{}".format(path,i))
        
        

def full_eval(SETTINGS):
  accuracy = {}
  #acc_best_single, recall,params = evaluate("{}/{}".format(SETTINGS["PATH"],"configuration.json"))
  for i in [5,15,25]:
    be = EnsembleManager("{}/{}".format(SETTINGS["PATH"],"configuration.json"),SETTINGS["DEVICES"][0])
    be.get_ensemble(i)
    accuracy["ensemble_{}".format(i)] = be.evaluate(2)
  #be.get_ensemble(10)
  #accuracy["distill"] = be.distill_model()
    
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
    return copy.deepcopy(self.config)

  def get_ratio(self):
    return self.offspring / self.evals



class model:
  def __init__(self,acc ,recall , config,SETTINGS,resamples):
    self.SETTINGS = SETTINGS
    self.ID = config["ID"]
    self.n_resamples = resamples
    self.mean_acc = acc
    self.config = config
    self.offspring = 0
    self.record_evals = 0
    self.cool = False
    self.flag_need_reload = True
    self.resampled = False
    self.evals = 5

  def load_scores_robust(self):
    path = "{}/{}/{}".format(self.SETTINGS["PATH"],"metrics",self.ID)
    try:
      self.df = pd.read_csv(path)
      if len(self.df) > self.n_resamples:
        self.resampled = True
      else:
        self.resampled = False

      if len(self.df) % self.n_resamples != 0:
        self.mu = np.nan
      else:
        self.mu = self.df["accuracy"].mean()
      self.sigma = self.df["accuracy"].std()
    except:
      self.resampled = True
      self.mu = np.nan

  def resampled_list(self):
    return self.resampled

  def load_scores(self):
    path = "{}/{}/{}".format(self.SETTINGS["PATH"],"metrics",self.ID)
    try:
      self.df = pd.read_csv(path)
      self.mu = self.df["accuracy"].mean()
      self.sigma = self.df["accuracy"].std()
    except:
      self.mu = self.mean_acc

  def sample(self):
    self.load_scores()
    return self.mu

  def sample_robust(self):
    self.load_scores_robust()
    return self.mu

  def sample_mu(self):
    if self.flag_need_reload:
      self.load_scores()
      if not self.cooldown():
        self.flag_need_reload = False
    return self.mu

  def get_config(self):
    self.offspring += 1 
    return self.config

  def get_ratio(self):
    return self.offspring / self.evals
  def set_cooldown(self):
    self.record_evals = self.evals
    self.flag_need_reload = True
  def cooldown(self):
    return self.record_evals == self.evals
  def update_eval(self):
    self.evals+=1#BROKEN!!

def main(worker, configspace: ConfigurationSpace, json_config):

    # ================= INITIALIZATION =================
    M_FORMAT = model
    # Load configurations and settings
    with open(json_config) as f:
        data = json.load(f)
        dataset_name = data["WORKER_CONFIG"]["DATASET_CONFIG"]["NAME"]
        resamples = data["WORKER_CONFIG"]["REPEAT"]
        SETTINGS = data["SEARCH_CONFIG"]

    # Setup essential variables
    TIME_SINCE_IMPROVE = 0
    EARLY_STOP = SETTINGS["EARLY_STOP"]
    START_TIME = time.time()
    RUNTIME = SETTINGS["RUNTIME"]
    last_print = time.time()
    active_parents = []
    train = train_eval(worker, json_config)

    history = []
    if SETTINGS["RESUME"]:
        data = load(data["EXPERIMENT_NAME"])
        history.extend([M_FORMAT(s, r, p, SETTINGS, resamples) for s, r, p in zip(data["scores"], data["recall"], data["config"])])
    else:
        configs = configspace.sample_configuration(SETTINGS["INITIAL_POPULATION_SIZE"])
        scores, recall, pop = train.init_async(configs)
        history.extend([M_FORMAT(s, r, p, SETTINGS, resamples) for s, r, p in zip(scores, recall, pop)])
        last_mean_best = None
        iteration = 0

    print("Initialization Complete.")

    # ================= PBT MAIN LOOP =================
    
    while iteration < SETTINGS["TOTAL_EVALUATIONS"]:

        # ---- WAIT FOR NEW RESULTS ----
        scores, recall, pop = [], [], []
        while len(scores) == 0:
            time.sleep(0.05)
            scores, recall, pop = train.get_async()
        history.extend([M_FORMAT(s, r, p, SETTINGS, resamples) for s, r, p in zip(scores, recall, pop)])

        # ---- THOMPSON SAMPLING ----
        scores = [i.sample() for i in history]
        top_10_indices = np.argsort(scores)[-10:][::-1]

        mean_best = max([i.sample() for i in history])
        if mean_best == 1:
            print("Achieved Perfect Score.")
            break

        # ---- EXPLOIT & EXPLORE (PBT Principle) ----
        while train.config_queue.qsize() < SETTINGS["CORES"]:
                # Here, we're considering the 'mutate_graph' as the method to slightly perturb the hyperparameters.
                train.update_async(configspace.mutate_graph(history[random.choice(top_10_indices)].get_config(), random.randint(0,10)))

        # ---- VERBOSE OUTPUTS ----
        if iteration % 10 == 0:
            print(f"[{dataset_name}] Generation: {iteration} -- Best (Mean) Score: {mean_best}")
            if len(history) > 10:
                print("Top 10 best models:")
                sorted_list = sorted(history, key=lambda obj: obj.sample(), reverse=True)
                for i in range(10):
                    print("ID: {} -- ACC: {}".format(sorted_list[i].ID, sorted_list[i].sample()))



        # ---- EARLY STOPPING & RUNTIME CHECK ----
        if time.time() > START_TIME + RUNTIME:
            print(f"Reached Total Allotted Time: {RUNTIME}")
            break
        elif (time.time() - last_print) > 30:
            print(f"TIME LEFT: {START_TIME + RUNTIME - time.time()} SECONDS")
            last_print = time.time()

        if mean_best == last_mean_best:
            TIME_SINCE_IMPROVE += len(pop)
        else:
            TIME_SINCE_IMPROVE = 0
        last_mean_best = copy.copy(mean_best)
        if TIME_SINCE_IMPROVE > EARLY_STOP:
            print("Early Stopping Triggered.")
            break
        iteration += 1

    print("PBT Main Loop Completed.")
    
    # ================= CLEANUP =================
    
    train.kill_workers()
    full_eval(SETTINGS)
    clean_up_weights_end(data, active_parents, history)
   

if __name__ == "__main__":
  with open(sys.argv[1]) as f:
    DATA = json.load(f)
    full_eval(DATA["SEARCH_CONFIG"])
