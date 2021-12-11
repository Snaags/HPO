import multiprocessing
import time
import csv
from pynvml import *
def assign_gpu():
  nvmlInit()
  max_memory = 6000000000
  count = nvmlDeviceGetCount()  
  gpu_list = []
  for i in range(count):
    h = nvmlDeviceGetHandleByIndex(i)
    info = nvmlDeviceGetMemoryInfo(h)
    free_memory = info.free
    print(free_memory)
    gpu_list.append(0)
    while free_memory > max_memory:
      free_memory -= max_memory
      gpu_list[-1] += 1
  print("GPU list is ",str(gpu_list))
  nvmlShutdown()
  return gpu_list

class train_eval:
  def __init__(self, worker, num_worker, filename):
    
    self.config_queue = multiprocessing.Queue()
    self.gpu_slots = multiprocessing.Queue()
    self.num_worker = num_worker
    self.results = multiprocessing.Queue()
    self.acc_list_full = []
    self.recall_list_full = []
    self.config_list_full = []
    self.filename = filename
    self.worker = worker

  def eval(self, population ):
    self.acc_list = []
    self.recall_list = []
    self.config_list = []
       
    self.processes = []
    gpu = assign_gpu()
    
    for i in population:
      self.config_queue.put(i.get_dictionary())
    #Initialise GPU slots
    for i in gpu:
      slots = i
      idx = gpu.index(slots)
      while slots != 0:
        self.gpu_slots.put(idx)
        slots  -= 1 
        gpu[idx] = slots
    for i in range(self.num_worker):
      print("Number of workers: {}".format(self.num_worker))
      self.processes.append(multiprocessing.Process(target = self.worker , args = (i, self.config_queue , self.gpu_slots, self.results)))
  
    ###Main Evaluation loop
    for i in self.processes:
      i.start()
    while not self.config_queue.empty():
      time.sleep(20)
      self.write2file()  
     
    for i in self.processes:
      i.join()    
      i.close() 
    self.write2file()
    return self.match_output_order_to_input(population)
 
  def match_output_order_to_input(self, in_pop):
    out_acc = []
    out_recall = []
    in_pop_dict_list = []
    for i in in_pop:
      in_pop_dict_list.append(i.get_dictionary())
    print("Length of pop {}".format(len(in_pop)))
    print("Length of accuracy list  {}".format(len(self.acc_list)))
    print("Length of accuracy full list  {}".format(len(self.acc_list_full)))
    for i in self.config_list:
      out_acc.append(self.acc_list[in_pop_dict_list.index(i)])
      out_recall.append(self.recall_list[in_pop_dict_list.index(i)]) 
    return out_acc , out_recall , in_pop_dict_list
       
 
  def write2file(self):
    while not self.results.empty():
      out = self.results.get()
      self.acc_list.append(out[1])
      self.recall_list.append(out[2])
      self.config_list.append(out[0])
      self.acc_list_full.append(out[1])
      self.recall_list_full.append(out[2])
      self.config_list_full.append(out[0])
      print("Number of models evaluated: ", len(self.acc_list_full))
      with open(self.filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        for acc , recall , config in zip(self.acc_list_full , self.recall_list_full , self.config_list_full):
          writer.writerow([acc, recall , config]) 
  

  



