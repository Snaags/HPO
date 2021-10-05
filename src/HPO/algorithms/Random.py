from multiprocessing import Pool
from HPO.utils.visualisation import plot_scores
from HPO.workers.repeat_worker import worker_wrapper
import csv
from ConfigSpace import ConfigurationSpace

def main(worker, configspace : ConfigurationSpace):
  TOTAL_EVALUATIONS = 500
  cores = 1
  pop = []
  scores = [] 
  worker = worker_wrapper(worker, 5)
  if cores == 1:
    results = []
    completed_configs = []
    for i in range(TOTAL_EVALUATIONS):
      config = configspace.sample_configuration()
      results.append(worker(config))
      completed_configs.append(config)
      with open("Random.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for config,score in zip(completed_configs, results):
          writer.writerow([score, config]) 

  else:
    population = configspace.sample_configuration(TOTAL_EVALUATIONS)
    pop = []
    for i in population:
      pop.append(i.get_dictionary())
    with Pool(processes = cores) as pool:
        results = pool.map(worker, pop)
    for i in results:
      scores.append(i)
    with open("Random.csv", "w") as csvfile:
      writer = csv.writer(csvfile)
      for config,score in zip(pop, scores):
        writer.writerow([config,score]) 
  
  print("Best Score: ", max(scores))      
  plot_scores(scores)
  
  best_config = pop[scores.index(max(scores))]
  best_score_validate = worker.validate(best_config)   
  print(best_config)
  print(best_score_validate) 
 
