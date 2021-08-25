from multiprocessing import Pool
from utils.visualisation import plot_scores
def main(worker, configspace):
  TOTAL_EVALUATIONS = 50
  cores = 2
  pop = []
  scores = [] 
  population = configspace.sample_configuration(TOTAL_EVALUATIONS)
  pop = []
  for i in population:
      pop.append(i.get_dictionary())
  if cores == 1:
    results = []
    for i in pop:
      results.append(worker.compute(i))
  else:
    with Pool(processes = cores) as pool:
        results = pool.map(worker.compute, pop)
  for i in results:
      scores.append(i)
  
  print("Best Score: ", max(scores))      
  plot_scores(scores)
  
  best_config = pop[scores.index(max(scores))]
  best_score_validate = worker.validate(best_config)   
  print(best_config)
  print(best_score_validate) 
 
