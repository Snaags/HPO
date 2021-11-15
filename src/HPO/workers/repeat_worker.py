from typing import Callable
import numpy as np 
import torch
def worker_wrapper( worker : Callable , num_evaluations : int = 10) -> Callable:
  
  def transformed_worker(*args , **kwargs):
    metrics = []
    for i in range(num_evaluations):
      metrics.append( worker(*args , **kwargs) )
    metrics_arr = np.array(metrics)
    print(metrics_arr)
    print(np.mean(metrics_arr, axis = 0 ))
    return np.mean(metrics_arr, axis = 0 )


  return transformed_worker
      
  
