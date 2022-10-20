from HPO.utils.visualisation import plot_scores
import csv
from ConfigSpace import ConfigurationSpace
from HPO.algorithms.algorithm_utils import train_eval

def main(worker, configspace : ConfigurationSpace, filename = "random.csv"):
  
  TOTAL_EVALUATIONS = 500
  cores = 2
  
  train = train_eval( worker , cores , filename = filename)
  configs = configspace.sample_configuration(TOTAL_EVALUATIONS)
  scores ,recall , pop= train.eval(configs)
  print("Best Score: ", max(scores))      
  plot_scores(scores)
  
  best_score = max(scores)
  best_config = pop[scores.index(max(scores))]
  best_rec = recall[scores.index(max(scores))]
  print(best_config)
  return best_config, best_score, best_rec
 


