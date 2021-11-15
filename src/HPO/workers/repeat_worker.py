from typing import Callable
import numpy as np 
import torch
from HPO.data.datasets import Mixed_repsol_feature as Dataset_
from HPO.data.datasets import Mixed_repsol_full as Dataset_2
from HPO.data.datasets import Subset
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
      
  
def one_out_cv( worker : Callable) -> Callable:
  
  def transformed_worker(*args , **kwargs):
    full_dataset = Dataset_()
    metrics = []
    size = len(full_dataset)
    for count,i in enumerate(range(size-1)):
      idx = [x for x in range(size)]
      idx.pop(i)
      train = Subset(full_dataset , np.asarray(idx))
      test = Subset(full_dataset , np.asarray([i]))
      metrics.append( worker(*args , **kwargs, train_dataset = train , test_dataset = test) )

      print("Mean ACC and recall {}".format(np.nanmean(np.array(metrics), axis = 0 )))
      print("Iteration Number: {} / {}".format(count, size))
    metrics_arr = np.array(metrics)
    print(metrics_arr)
    return np.nanmean(metrics_arr, axis = 0 )




  return transformed_worker
def one_out_cv_aug(augs = 0 , n_test = 1):
  def wrapper(worker : Callable) -> Callable:
    def transformed_worker(*args , **kwargs):
      full_dataset = Dataset_2(augs)
      metrics = []
      size = len(full_dataset)
      for count, i in enumerate(full_dataset.aug_source_dict):
        remove_list = []
        test_list = []
        keys = list(full_dataset.aug_source_dict.keys())
        x =  keys[keys.index(i)]
        remove_list += [x] + full_dataset.aug_source_dict[x]
        test_list += [x]
        idx = [x for x in range(size)]
  
        train_list = [idx[e] for e,i in enumerate(idx) if i not in remove_list]
        
        train = Subset(full_dataset , np.asarray(train_list))
        test = Subset(full_dataset , np.asarray(test_list))
        metrics.append( worker(*args , **kwargs, train_dataset = train , test_dataset = test) )
        print("Mean ACC and recall {}".format(np.nanmean(np.array(metrics), axis = 0 )))
        print("Iteration Number: {} / {}".format(count, len(full_dataset.aug_source_dict.keys())))
      metrics_arr = np.array(metrics)
      print(metrics_arr)
      return np.nanmean(metrics_arr, axis = 0 )
  
  
    return transformed_worker
  return wrapper
