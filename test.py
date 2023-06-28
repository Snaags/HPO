import numpy as np

run_experiment(dataset,gpu,gpus):
  
  #do experiment code

  gpus[gpu] += 1 
  return

gpus = [1,1,1,1]

for i in dataets:
  while max(gpus) == 0:
    time.wait()
  gpu = np.argmax(gpus)
  gpus[gpu] -= 1 
  run_experiment(i,gpu,gpus)#assume this is done in background
