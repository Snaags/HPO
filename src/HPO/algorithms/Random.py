from HPO.utils.visualisation import plot_scores
from HPO.workers.repeat_worker import worker_wrapper
import csv
from ConfigSpace import ConfigurationSpace
from HPO.algorithms.algorithm_utils import train_eval

def main(worker, configspace : ConfigurationSpace):
  
  TOTAL_EVALUATIONS = 1000
  cores = 30
  train = train_eval( worker , cores , filename = "Random.csv")
  configs = configspace.sample_configuration(TOTAL_EVALUATIONS)
  scores ,recall , pop= train.eval(configs)
  print("Best Score: ", max(scores))      
  plot_scores(scores)
  
  
  best_config = pop[scores.index(max(scores))]
  best_score_validate = worker.validate(best_config)   
  print(best_config)
  print(best_score_validate) 
 
