from typing import Callable
import numpy as np 
def worker_wrapper( worker : Callable , num_evaluations : int ) -> Callable:
  
  def transformed_worker(*args , **kwargs):
    metrics = []
    for i in range(num_evaluations):
      metrics.append( worker(*args , **kwargs) )
    metrics_arr = np.array(metrics)
    print(metrics_arr)
    return np.mean(metrics_arr, axis = 0 )

  return transformed_worker
      
  
