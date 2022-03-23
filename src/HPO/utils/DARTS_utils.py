import copy
from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')



def config_space_2_DARTS(hyperparameters,reduction = False):
  normal = [0,0,0,0,0,0,0,0]
  reduce = [0,0,0,0,0,0,0,0]
  for i in hyperparameters:
    if "normal" in i:
      idx = (int(i[-3])*2) + int(i[-1]) 
      if "index" in i:
        if type(normal[idx]) == int:
          normal[idx] = [0,hyperparameters[i]]
        else:
          normal[idx][1] = hyperparameters[i]
        
      if "node" in i:
        if type(normal[idx]) == int:
          normal[idx] = [hyperparameters[i], 0]
        else:
          normal[idx][0] = hyperparameters[i]
           
    if "reduction" in i:
      idx = (int(i[-3])*2) + int(i[-1]) 
      idx = (int(i[-3])*2) + int(i[-1]) 
      if "index" in i:
        if type(reduce[idx]) == int:
          reduce[idx] = [0,hyperparameters[i]]
        else:
          reduce[idx][1] = hyperparameters[i]
        
      if "node" in i:
        if type(reduce[idx]) == int:
          reduce[idx] = [hyperparameters[i], 0]
        else:
          reduce[idx][0] = hyperparameters[i]
  normal_concat = [x for x in range(2,len(normal)//2+2)]
  reduce_concat = [x for x in range(2,len(normal)//2+2)]
  if reduction == False:
    print("Reduction cell architecture disabled!")
    reduce = copy.copy(normal)
  for n,r in zip(normal, reduce):
    if n[1] in normal_concat:
      normal_concat.pop(normal_concat.index(n[1]))
    if r[1] in reduce_concat:
      reduce_concat.pop(reduce_concat.index(r[1]))
  normal = [tuple(x) for x in normal ]
  reduce = [tuple(x) for x in reduce ]
  return Genotype(normal , normal_concat, reduce, reduce_concat)
