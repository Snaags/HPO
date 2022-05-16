from HPO.utils.visualisation import plot_scores
import csv
from ConfigSpace import ConfigurationSpace
from HPO.algorithms.algorithm_utils import train_eval

def main(worker, configspace : ConfigurationSpace):
  
  TOTAL_EVALUATIONS = 64
  cores = 16

  train = train_eval( worker , cores , filename = "Random.csv")
  configs = configspace.sample_configuration(TOTAL_EVALUATIONS)
  scores ,recall , pop= train.eval(configs)
  print("Best Score: ", max(scores))      
  plot_scores(scores)
  
  
  best_config = pop[scores.index(max(scores))]
  print(best_config)
 


