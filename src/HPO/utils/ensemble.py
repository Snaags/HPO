import torch
from HPO.utils.model_graph import ModelGraph
import torch.nn.functional as F
from HPO.data.UEA_datasets import UEA_Train, UEA_Test, UEA_Full
import time
import matplotlib.pyplot as plt
from HPO.utils.FCN import FCN 
import pandas as pd
import torch
from HPO.data.teps_datasets import Train_TEPS , Test_TEPS
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import HPO.utils.augmentation as aug
from HPO.utils.train_utils import collate_fn_padd
from HPO.utils.train import train_model
from queue import Empty
from sklearn.model_selection import StratifiedKFold as KFold
from collections import namedtuple
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
import csv
from HPO.data.dataset import get_dataset
import HPO.utils.augmentation as aug

class EnsembleManager:
        def __init__(self , JSON_CONFIG,device):
                with open(JSON_CONFIG,"r") as f:
                        data = json.load(f)
                self.path = JSON_CONFIG[:-18]
                self.cuda_device =device
                self.accuracy,  self.recall, self.configs = self.load_hps(self.path)
                self.SETTINGS = data["WORKER_CONFIG"]
                self.channels = data["ARCHITECTURE_CONFIG"]["STEM_SIZE"][0]
                name = self.SETTINGS["DATASET_CONFIG"]["NAME"]
                if data["GENERATE_PARTITION"]:
                        DS_PATH = self.SETTINGS["DATASET_CONFIG"]["DATASET_PATH"]
                else:
                        DS_PATH = None
                #if "AUGMENTATIONS" in self.SETTINGS:
                #        augs = aug.initialise_augmentations(self.SETTINGS["AUGMENTATIONS"])
                #else: 
                augs = None

                train_args = {"cuda_device":device,"augmentation" : None, "binary" :self.SETTINGS["BINARY"],"path" : DS_PATH}
                test_args = {"cuda_device":device,"augmentation" :None, "binary" :self.SETTINGS["BINARY"],"path" : DS_PATH}
                train_dataset, self.test_dataset = get_dataset(name,train_args, test_args)
                # = UEA_Test(name = self.SETTINGS["DATASET_CONFIG"]["NAME"],device = 0)
                self.num_classes = self.test_dataset.get_n_classes()
                self.num_features = self.test_dataset.get_n_features()
                self.models = nn.ModuleList()

        def evaluate_old(self, batch_size):
                self.testloader = torch.utils.data.DataLoader(
                      self.test_dataset,collate_fn = collate_fn_padd,shuffle = True,
                      batch_size= batch_size ,drop_last = True)
                self.labels = np.zeros(shape = len(self.test_dataset))
                self.preds = np.zeros(shape = len(self.test_dataset))
                self.ensemble.eval()
                for index, (x,y) in enumerate(self.testloader):
                        self.labels[ index*batch_size :(index+1)*batch_size] = y.detach().cpu().numpy()
                        self.preds[ index*batch_size :(index+1)*batch_size] = self.ensemble(x.float(),y).detach().cpu().numpy()
                self.confusion_matrix = confusion_matrix(self.labels,self.preds,labels = list(range(self.num_classes)))
                with np.printoptions(linewidth = (10*self.num_classes+20),precision=4, suppress=True):
                    print(self.confusion_matrix)
                correct = np.sum(np.diag(self.confusion_matrix))
                total = np.sum(self.confusion_matrix)
                print("Accuracy: {}".format(correct/total))



        def evaluate(self, batch_size):
            self.testloader = torch.utils.data.DataLoader(
                self.test_dataset, collate_fn=collate_fn_padd, shuffle=True, 
                batch_size=batch_size, drop_last=True)

            labels_list = []
            preds_list = []
            self.ensemble.eval()
            
            for index, (x, y) in enumerate(self.testloader):
                x, y = x.float(), y  # moving to the device
                labels_list.append(y.detach())  # append the tensor to the list directly
                preds_list.append(self.ensemble(x, y).detach())  # append the tensor to the list directly

            # concatenate all tensors along the 0-th dimension
            labels = torch.cat(labels_list).cpu().numpy()
            preds = torch.cat(preds_list).cpu().numpy()

            self.confusion_matrix = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
            with np.printoptions(linewidth=(10 * self.num_classes + 20), precision=4, suppress=True):
                print(self.confusion_matrix)
            correct = np.sum(np.diag(self.confusion_matrix))
            total = np.sum(self.confusion_matrix)
            print("Accuracy: {}".format(correct / total))
            return correct / total


        def get_ensemble(self,n_classifiers = 10):
                order = list(np.argsort(self.accuracy))
                """
                weights = os.listdir("{}weights/".format(self.path))
                accs = [ float(i.split("-")[-1]) for i in weights]
                order =  list(np.argsort(accs))
                """

                while len(self.models) < n_classifiers:
                        index = order.pop(-1)
                        print("Model acc: {}".format(self.accuracy[index]))
                        m, sucess = self.try_build(index)
                        if sucess:
                          self.models.extend(m)
                self.ensemble = Ensemble(self.models,self.num_classes,self.cuda_device)


        def load_hps(self, PATH,FILENAME = "evaluations.csv"):
            scores = []
            recall = []
            config = []
            with open( "{}{}".format(PATH,FILENAME) , newline = "") as csvfile:
                    reader = csv.reader(csvfile, delimiter = ",")
                    for row in reader:
                        scores.append(float(row[0]))
                        recall.append(float(row[1]))
                        config.append(eval("".join(row[2])))
            return scores, recall, config
    
        def find_all(self,acc):
            current_val = []
            for idx,i in enumerate(self.accuracy):
                if acc == i:
                        current_val.append(idx)
            return current_val

        def try_build(self,index):
            hyperparameter = self.configs[index]
            print(hyperparameter)
            ID = hyperparameter["ID"]
            weights = os.listdir("{}weights/".format(self.path))
            num_len = len(str(ID))
            models = []
            for inst in weights:
              if int(inst.split("-")[0]) == int(ID):
                print(inst,ID)
                match = inst

                state = torch.load("{}weights/{}".format(self.path,match))
                if "graph" in hyperparameter.keys():
                        print("loading graph")
                        models.append(ModelGraph(self.test_dataset.get_n_features(),self.channels,self.test_dataset.get_n_classes(),
                          self.test_dataset.get_length(),hyperparameter["graph"],hyperparameter["ops"],device = self.cuda_device,
                          binary = self.SETTINGS["BINARY"],dropout = self.SETTINGS["DROPOUT"],droppath = self.SETTINGS["DROPPATH"],
                          raw_stem = self.SETTINGS["RAW_STEM"],embedding = self.SETTINGS["EMBEDDING"]))
                        models[-1].load_state_dict(state)
                else: 
                        print("loading cell")
                        gen = config_space_2_DARTS(hyperparameter, reduction = True)
                        #print("Loaded Weights")
                        #print("Weight mismatch")
                        models.append(NetworkMain(self.num_features,
                                    2**hyperparameter["channels"], num_classes= self.num_classes,
                                  layers = hyperparameter["layers"], auxiliary = False,
                                  drop_prob = self.SETTINGS["P"],genotype = gen, 
                                  binary = self.SETTINGS["BINARY"]).load_state_dict(state))
            return models,True

        def load_state(path,ID):
                state = torch.load("{}{}".format(path,ID))
        


class Ensemble(nn.Module):
        def __init__( self, models,num_classes ,device):
                super(Ensemble,self).__init__()
                
                self.num_classes = num_classes
                self.classifiers = models.cuda(device)

        def soft_voting( self, prob ):
                #SUM PROBABILITIES FROM ALL CLASSIFIERS THEN GET MAX PROBABILITY DENSITY
                t_prob = torch.sum(prob, axis =  2)
                preds = torch.argmax(t_prob,dim = 1)
                return preds #shape [batch_size] 
        def hard_voting(self, prob):
                #GET PREDICTION FROM EACH MODEL THEN CALCULATE MODE (MAJORITY)
                preds = torch.argmax(prob , axis = 1)
                return torch.mode(preds,dim = 1)[0] #Shape [batch_size]

        def forward( self, x ,y):
                self.batch_size = x.shape[0]
                probs = torch.zeros(size = (self.batch_size ,self.num_classes, len(self.classifiers) ))
                for idx , model in enumerate(self.classifiers):
                        probs[:, :, idx] = F.sigmoid(model(x))
                #print(probs)
                #print(F.sigmoid(probs))
                #print(y)
                return self.soft_voting(probs)

        def eval(self):
                for i in self.classifiers:
                        i.eval()

if __name__ == "__main__":
	import sys
	be = EnsembleManager(sys.argv[1],0)
	be.get_ensemble(1)
	be.evaluate(2)
