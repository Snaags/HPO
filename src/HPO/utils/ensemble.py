import torch
import numpy as np 
import torch.nn as nn
import os
import sys
import copy
from HPO.utils.model import NetworkMain
from sklearn.metrics import confusion_matrix
from HPO.utils.DARTS_utils import config_space_2_DARTS
import json
from HPO.data.UEA_datasets import UEA_Train, UEA_Test, UEA_Full

class EnsembleManager:
	def __init__(self , JSON_CONFIG):
		self.accuracy,  self.recall, self.configs = self.load_hps(path)
		with open(JSON_CONFIG,"w") as f:
			data = json.load(f)
			self.path = data["PATH"]
		self.SETTINGS = data["WORKER_CONFIG"]
		self.test_dataset = UEA_Test(name = self.SETTINGS["NAME"])
		self.num_classes = self.test_dataset.get_n_classes()
		
		self.models = nn.ModuleList()

	def evaluate(self, batch_size):
		self.testloader = torch.utils.data.DataLoader(
                      self.test_dataset,collate_fn = collate_fn_padd,shuffle = True,
                      batch_size= batch_size ,drop_last = True)
		self.labels = np.zeros(shape = len(self.test_dataset))
		self.preds = np.zeros(shape = len(self.test_dataset))
		for index, (x,y) in enumerate(self.testloader):
			self.labels[ index*batch_size :(index+1)*batch_size] = y
			self.preds[ index*batch_size :(index+1)*batch_size] = self.ensemble(x)
		self.confusion_matrix = confusion_matrix(self.labels,self.preds,labels = list(range(self.num_classes)))
		with np.printoptions(linewidth = (10*self.n_classes+20),precision=4, suppress=True):
            print(self.confusion_matrix)
    	correct = np.sum(np.diag(self.confusion_matrix))
    	total = np.sum(self.confusion_matrix)
    	print("Accuracy: {}".format(correct/total))


	def get_ensemble(self,n_classifiers = 10):
		acc_temp = copy.copy(self.accuracy)
		for i in range(n_classifiers):
			acc = max(acc_temp)
			self.models.append(self.try_build(acc))
			acc_temp.remove(acc)
		self.ensemble = Ensemble(self.models,self.num_classes)


	def load_hps(self, PATH,FILENAME = "evaluations.csv"):
    	scores = []
    	recall = []
    	config = []
    	with open( "{}{}".format(PATH,FILENAME) , newline = "") as csvfile:
        	reader = csv.reader(csvfile, delimiter = ",")
        	for row in reader:
            	scores.append(float(row[0]))
            	recall.append(float(row[1]))
            	config.append(eval("".join(row[2:])))
    	return acc, recall, config

    def find_all(self,acc):
    	current_val = []
    	for idx,i in enumerate(self.accuracy):
    		if acc == i:
    			current_val.append(idx)
    	return current_val

    def try_build(self, acc):
    	state = torch.load(self.path+acc)
    	current_val = self.find_all(acc)
    	for idx in current_val:
    	hyperparameter = self.configs[index]
	    	gen = config_space_2_DARTS(hyperparameter)
			model = NetworkMain(self.num_features,
				2**hyperparameter["channels"], num_classes= self.num_classes,
				layers = hyperparameter["layers"], auxiliary = False,drop_prob = self.SETTINGS["P"], 
				genotype = gen, binary = self.SETTINGS["BINARY"])
			if len(model.load_state_dict(state)[0]) == 0:
				break
		return model

	def load_state(path,ID):
		state = torch.load("{}{}".format(path,ID))
	


class Ensemble(nn.Module):
	def __init__( self, models,num_classes ):
		self.num_classes = num_classes
		self.classifiers = nn.ModuleList()
		super(Ensemble).__init__()

	def soft_voting( self, prob ):
		#SUM PROBABILITIES FROM ALL CLASSIFIERS THEN GET MAX PROBABILITY DENSITY
		t_prob = torch.sum(prob, axis =  2)
		preds = torch.argmax(t_prob,dim = 1)
		return preds #shape [batch_size] 
	def hard_voting(self, prob):
		#GET PREDICTION FROM EACH MODEL THEN CALCULATE MODE (MAJORITY)
		preds = torch.argmax(probs , axis = 1)
		return torch.mode(preds,dim = 1)[0] #Shape [batch_size]

	def forward( self, x ):
		self.batch_size = x.shape[0]
		probs = torch.zeros(size = (self.batch_size ,self.num_classes, self.classifier ))
		for idx , model in enumerate(self.classifiers):
			probs[:, :, idx] = model(x)
		return self.soft_voting(probs)

if __name__ == "__main__":
	import sys
	be = BuildEnsemble(sys.argv[1])
	ensemble_model = be.get_ensemble(10)
	for x,y in 