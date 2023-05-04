import importlib
import time
from multiprocessing import Pool
from HPO.utils.seed import set_seed
import numpy as np
import logging
import sys
import os
import json
from datetime import datetime
import sqlite3
import cProfile

def main(JSON_CONFIG):
  with open(JSON_CONFIG) as conf:
    data = json.load(conf)
  if not data["SEARCH_CONFIG"]["RESUME"]:
    if not data["EXPERIMENT_NAME"]:
      #CREATE EXPERIMENT DIRECTORY AND LOAD JSON DATA
      name = "".join(str(datetime.now()).split(" "))
    else: 
      name = data["EXPERIMENT_NAME"]
    if os.path.exists("experiments/{}".format(name)):
      raise Exception("Experiment name exists, rename current experiment")

    os.system("mkdir experiments/{}".format(name))
    os.system("mkdir experiments/{}/weights".format(name))
  else:
    name = data["EXPERIMENT_NAME"]
  #IMPORT MODULES
  _worker = importlib.import_module("HPO.workers.{}".format(data["WORKER_MODULE_NAME"])) 
  _config =importlib.import_module("HPO.searchspaces.spaces") 
  _algorithm = importlib.import_module("HPO.algorithms.{}".format(data["SEARCH_MODULE_NAME"])) 
  #DEFINE FUNCTIONS FROM MODULES
  algorithm = _algorithm.main
  worker = _worker.compute



  config = eval("_config.{}({})".format(data["CONFIG_MODULE_NAME"],data))
  #STORE DATA IN EXPERIMENT JSON FILE 
  data["DATE"] = "".join(str(datetime.now()).split(" "))
  data["START_TIME"] = time.time()
  data["SEARCH_CONFIG"]["PATH"] = "experiments/{}".format(name)
  data["SEARCH_SPACE"] = str(config)
  experiment_json = "experiments/{}/configuration.json".format(name)
  with open(experiment_json,"w") as f:
    json.dump(data,f, indent=4)
  logging.basicConfig(filename='experiments/{}/experiment.log'.format(name),  level=logging.DEBUG)
  time.sleep(1) 
  set_seed(experiment_json)
  #START SEARCH

  # Connect to the SQLite database (this will create a new file called 'training_info.db')
  if data["WORKER_CONFIG"]["LOGGING"]:
    database = data["DATABASE_NAME"]
    conn = sqlite3.connect(database)
    c = conn.cursor()

    # Create a table to store training information

    # Create the training_info table
    c.execute("""
    CREATE TABLE IF NOT EXISTS training_info (
        experiment TEXT,
        dataset TEXT,
        model_id INTEGER,
        epoch INTEGER,
        training_loss REAL,
        validation_loss REAL,
        training_accuracy REAL,
        validation_accuracy REAL,
        learning_rate REAL,
        confusion_matrix_train TEXT,
        confusion_matrix_test TEXT,
        fold INTEGER,
        repeat INTEGER,
        parameters INTEGER,
        PRIMARY KEY (model_id, epoch)
    )
    """)
    conn.commit()

  if False:
    def profiled_worker(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = worker(*args, **kwargs)
        
        profiler.disable()
        profiler.dump_stats(f'profile_output_{os.getpid()}.txt')
        
        return result
    algorithm(profiled_worker, config,experiment_json)
  else:
    algorithm(worker, config,experiment_json)
  with open(JSON_CONFIG) as conf:
    data = json.load(conf)
  data["END_TIME"] = time.time()
  with open(JSON_CONFIG,"w") as f:
    json.dump(data,f,indent=4)


if __name__ == "__main__":

  #PASS JSON CONFIGURATION FILE NAME TO MAIN
  if os.path.isdir(sys.argv[1]):
    files = os.listdir(sys.argv[1])  
    for runs in files:
      main("{}/{}".format(sys.argv[1],runs))
    
  else:
    main(sys.argv[1])

  

def meta_cv():
  from HPO.algorithms.meta_cv import MetaCV
  from HPO.workers.repsol_validate import _compute as validate
  ##Hyperparameters for NAS 
  hpo = {
    "batch_size" : 2,
    "channels" : 27,
    'jitter': 0.12412584247629389, 'jitter_rate': 0.5439942968995378, 'mix_up': 0.19412584247629389, 'mix_up_rate': 0.5439942968995378,
    'cut_mix': 0.19412584247629389, 'cut_mix_rate': 0.5439942968995378,'cut_out': 0.09412584247629389, 'cut_out_rate': 0.7439942968995378,
    'crop': 0.19412584247629389, 'crop_rate': 0.5439942968995378,
    'scaling': 0.001317169415702424, 'scaling_rate': 0.43534309734597858, 'window_warp_num': 3, 'window_warp_rate': 1.40015481616041954,
    'lr': 0.005170869707739693, 'p': 0.00296905723528657, 
    "epochs" : 70,
    "layers" : 3}
  hpo = {
    "batch_size" : 2,
    "channels" : 27,
    'lr': 0.005170869707739693, 'p': 0.0, 
    "epochs" : 50,
    "layers" : 3}
  ##Settings 
  NUM_FOLDS = 10
  budget = 8
  logging.info("Starting MetaCV")
  metacv = MetaCV(source_path = "/home/cmackinnon/scripts/datasets/repsol_mixed", destination_path = "/home/cmackinnon/scripts/datasets/repsol-meta-cv", num_samples = 87, num_folds = NUM_FOLDS)
  for i in range(NUM_FOLDS):
    metacv.next_fold()
    best_config, best_score , best_rec= algorithm(worker, config, "reg_evo_{}.csv".format(metacv.current_fold)) 
    print("Validated results for meta CV fold {}:".format(i))
    logging.info("Validated results for meta CV fold {}:".format(i))
    for top , (hpo_score,hpo_rec,hpo_config) in enumerate(zip(best_score,best_rec,best_config)):
      hpo_config.update(hpo)
      with Pool(budget) as pool:
        res = pool.starmap(validate, [[hpo_config,1]]*8)
      acc = np.mean(np.asarray(res)[:,0])
      rec = np.mean(np.asarray(res)[:,1])
    
      print("HPO CV score [TOP {} model] - ACC: {} - REC: {}".format(top,hpo_score,hpo_rec))
      logging.info("HPO CV score [TOP {} model] - ACC: {} - REC: {}".format(top,hpo_score,hpo_rec))
      print("MetaCV test set score [TOP {} model] - ACC: {} - REC: {}".format(top,acc,rec))
      logging.info("MetaCV test set score [TOP {} model] - ACC: {} - REC: {}".format(top,acc,rec))
