import torch
import types
from multiprocessing import Pool


class ResourceDevice:
	def __init__(self):
		pass


class CudaDevice(ResourceDevice):
	def __init__(self, device_id):
		self.device_id = device_id
		self.total_memory = torch.cuda.get_device_properties(device_id).total_memory
		self.reserved = torch.cuda.memory_reserved(device_id)
		self.allocated = torch.cuda.memory_allocated(device_id)
		self.free_reserved = r-a  # free inside reserved
		self.cores

class ResourceManager:
	def __init__(self, devices):
		"""
		Abstract class which should be used to build classes for 
		allocating and recording the different resources required by the workers
		"""
		self.true_free_resource

	def check_allocation(self, estimate_allocation_quantity : int):

		if allocation_quantity < self.true_free_resource:

			return True


	def check_availible_resources(self):
		pass



class CudaMemoryManager(ResourceManager):
	def __init__(self):
		self.num_gpu

	def Monitor(self):
		pass

class algorithm:


	def __init__(self,worker):
	
		#Sets the worker based on whether it is a function or a class	
		if isinstance(worker, types.FunctionType):
			self.worker = worker
		else:
			self.worker = worker().compute

		self.resource_list = self.init_resource_list()


		self.num_gpus = torch.cuda.device_count()

	def init_resource_list(self): 

		self.init_gpu_manager()

	def resource_check(self):
		pass
	def gpu_device_memory_check(self, device_id):
		pass	
	def allocate_gpu(self):
		pass
	def estimate_model_memory(self):
		pass
	def train_and_evaluate(self,configurations):
		pass
		

		#with Pool(self.cores, maxtasksperchild =  1 ) as p:
			

		#Should handle allocation to gpu

	def _train(self, config, device_id):
		pass

	def random_architecture_sample(self, cs, num : int = 1) -> list:
		if num > 1:
			return cs.sample_configuration(num)
		else:
			return [cs.sample_configuration()]

	def log_results(self, scores : list, configurations : list):
		#scores - list of all the scores upto this point
		#
		#
		#
		pass
		
