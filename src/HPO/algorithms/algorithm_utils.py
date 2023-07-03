from multiprocessing import Queue, Process
import time
import csv
import json
from pynvml import *
import cProfile

def profiled_function(worker,*args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = worker(*args, **kwargs)
    
    profiler.disable()
    profiler.dump_stats(f'profile_output_{os.getpid()}.txt')
    
    return result

def load(FILENAME):
    scores = []
    recall = []
    config = []
    params = []
    ID = []
    with open( "{}".format(FILENAME) , newline = "") as csvfile:
        reader = csv.reader(csvfile, delimiter = ",")
        for row in reader:
            scores.append(float(row[0]))
            recall.append(float(row[1]))
            config.append(eval("".join(row[2])))
            ID.append(config[-1]["ID"])
            if len(row) == 4:
               params.append(int(row[3])) 
    error = [1-x for x in scores]
    e_min = 1
    best_list = []
    for i in error:
      if i < e_min:
        e_min = i
      best_list.append(e_min)
    return {"accuracy":scores,"recall":recall,"config":config ,"params": params, "ID": ID}

def assign_gpu(devices):
  nvmlInit()
  max_memory = 1000000000
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
  for e,i in enumerate(gpu_list):
    if not (e in devices):
      gpu_list[e] = 0
  return gpu_list

class train_eval:
  def __init__(self, worker ,json_config):
    with open(json_config) as f:
      SETTINGS = json.load(f)["SEARCH_CONFIG"]
    if not os.path.exists("{}/metrics".format(SETTINGS["PATH"])):
        os.mkdir("{}/metrics".format(SETTINGS["PATH"]))
    self.num_worker = SETTINGS["CORES"]
    self.filename = "{}/{}".format(SETTINGS["PATH"],SETTINGS["FILE_NAME"])
    self.JSON_CONFIG = json_config
    self.config_queue = Queue()
    self.gpu_slots = Queue()
    self.devices = SETTINGS["DEVICES"]
    self.results = Queue()
    self.acc_list_full = []
    self.recall_list_full = []
    self.param_list_full= []
    self.config_list_full = []
    self.worker = worker
    if SETTINGS["RESUME"] == True and os.path.exists(self.filename):
      self.resume()
    else:
      self.ID_INIT = 0

  def resume(self):
    data_dict = load(self.filename)
    self.acc_list_full = data_dict["accuracy"]
    self.recall_list_full = data_dict["recall"]
    self.config_list_full = data_dict["config"]
    self.param_list_full=data_dict["params"]
    self.ID_INIT = max(data_dict["ID"])
    print("LOADING PREVIOUS DATA: ",self.acc_list_full)

  def allocate_gpu(self):
    if max(self.gpu) == 0:
      return None
    idx = self.gpu.index(max(self.gpu))
    self.gpu[idx] -= 1
    return idx
    

  def init_async(self, population):
    self.acc_list = []
    self.recall_list = []
    self.param_list = []
    self.config_list = []
    self.processes = []
    self.initial_pop_size = len(population)
    gpu = assign_gpu(self.devices)
    self.gpu =gpu
    for ID,i in enumerate(population):
      if type(i) != dict:
        c = i.get_dictionary()
      else:
        c = i
      if not "ID" in c:
        c["ID"] = len(self.config_list_full) + ID + self.ID_INIT
      self.config_queue.put(c)
    
    #Initialise GPU slots
    while True:
      slot = self.allocate_gpu()
      if slot == None:
        break
      else:
        self.gpu_slots.put(slot)
    #Initialise Processes
    i = 0
    while len(self.processes) < self.num_worker:
      self.processes.append(Process(target = self.worker , args = (i, self.config_queue , self.gpu_slots, self.results,self.JSON_CONFIG)))
      i+= 1
    ###Main Evaluation Loop###
    for i in self.processes:
      i.start()
    return [],[],[]

  def update_async(self,config):
    self.alive = [1] * self.num_worker
    for idx,i in enumerate(self.processes):
      if not i.is_alive():
        self.alive[idx] = 0
        self.gpu_slots.put(random.choice(self.devices))\
        self.processes[idx] = Process(target = self.worker , args = (idx, self.config_queue , self.gpu_slots, self.results,self.JSON_CONFIG))
        self.processes[idx].start()
      else:
        self.alive[idx] = 1
    #print("Number of Processes 'Alive': {}/{}".format(sum(self.alive)/len(self.alive)))
    """
    while len(self.processes) < self.num_worker:
      self.processes[idx] = Process(target = self.worker , args = (idx, self.config_queue , self.gpu_slots, self.results,self.JSON_CONFIG))
      self.processes[idx].start()
    """
    if type(config) != dict:
      c = config.get_dictionary()
    else:
      c = config
    if not "ID" in c:
      self.initial_pop_size += 1
      c["ID"] = self.initial_pop_size + self.ID_INIT
    self.config_queue.put(c)

  def get_async(self):
    self.acc_list = []
    self.recall_list = []
    self.config_list = []
    self.write2file()
    self.alive = [1] * self.num_worker
    for idx,i in enumerate(self.processes):
      if not i.is_alive():
        self.alive[idx] = 0

        self.processes[idx] = Process(target = self.worker , args = (idx, self.config_queue , self.gpu_slots, self.results,self.JSON_CONFIG))
        self.processes[idx].start()
      else:
        self.alive[idx] = 1
    return self.acc_list, self.recall_list, self.config_list

  def kill_workers(self):
    time.sleep(30)
    print("Post Join print")
    for e,i in enumerate(self.processes):
      i.join(5)
      i.terminate()
    print("Post close print")


  def eval(self, population, datasets = None):
    self.acc_list = []
    self.recall_list = []
    self.param_list = []
    self.config_list = []
    self.processes = []
    gpu = assign_gpu(self.devices)
    self.gpu =gpu
    for ID,i in enumerate(population):
      if type(i) != dict:
        c = i.get_dictionary()
      else:
        c = i
      if not "ID" in c:
        c["ID"] = len(self.config_list_full) + ID + self.ID_INIT

      self.config_queue.put(c)
    
    #Initialise GPU slots
    while True:
      slot = self.allocate_gpu()
      if slot == None:
        break
      else:
        self.gpu_slots.put(slot)
    #Initialise Processes
    for i in range(self.num_worker):
        print("Number of workers: {}".format(self.num_worker))
        self.processes.append(Process(target = self.worker , args = (i, self.config_queue , self.gpu_slots, self.results,self.JSON_CONFIG)))

    ###Main Evaluation Loop###
    for i in self.processes:
      i.start()
    while not self.config_queue.empty():
      time.sleep(10)
      self.write2file()  
     
    for i in self.processes:
      self.write2file()
      i.join() 
    self.write2file()
    return self.acc_list, self.recall_list, self.config_list#self.match_output_order_to_input(population)
 
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
        self.param_list.append(out[3])
        self.config_list.append(out[0])
        self.acc_list_full.append(out[1])
        self.recall_list_full.append(out[2])
        self.param_list_full.append(out[3])
        self.config_list_full.append(out[0])
        print("Number of models evaluated: ", len(self.acc_list_full))
        with open(self.filename, "a") as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow([out[1], out[2] , out[0], out[3]]) 


  



