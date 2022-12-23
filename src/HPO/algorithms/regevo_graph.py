import csv
import networkx as nx
import json
import math
import os 
import copy
import torch
from multiprocessing import Pool
from HPO.utils.graph_utils import gen_iter
import time
import collections
import random
import matplotlib.pyplot as plt 
import numpy as np
from ConfigSpace.configuration_space import Configuration
import ConfigSpace.util as csu
from HPO.utils.visualisation import plot_scores
from HPO.algorithms.algorithm_utils import train_eval 
###Tounament Selection/Aging Selection
  #
  #parameters:
  # S : (int) number of solutions to sample at each iteration
  # P : population size
  # C : Total cycles of evolution 
  # parent : highest-accuracy model in sample
  # history : Record of all models
  # population : Currently active models


# IDEAS
#
# - Run aging evolution for "species" with 1 species per core allowing cross-over occationally
#g
#
#
#
#
#

def F1(acc,rec):
    b = 1
    if acc != None and rec != None:
      acc += 0.0001
      rec += 0.0001
      
    return (1+b**2)+ ((acc*rec)/(((b**2)*(acc**-1))+(rec**-1)))

class Model:
  MUTATE_RATE = 0.2
  def __init__(self, cs,ops_list):
    self.graph = None
    self.op_list = ops_list
    self.ops = None 
    self.accuracy = None
    self.cs = cs
    self.recall = None

  def set_arch(self, arch):
    self._arch = arch
    self.graph = arch["graph"]
    self.ops = arch["ops"]
  def exchange_one( self , key  ):
    n_idx = []
    for idx ,i in enumerate(key):
      if i.isdigit():
        n_idx.append(idx)
    new_key = key[:n_idx[0]]+"1"+key[n_idx[-1]+1:]
    while True:
      new = self.cs.sample_configuration()[0]["ops"]
      if new[new_key] != self.ops[key]:
        print("Mutated {} to {}".format(self.ops[key],new[new_key]))
        self.ops[key] = new[new_key]
        return
  def generate_op(self):
    new = self.cs.sample_configuration()[0]["ops"]
    idx = len(self.graph) -1
    if "op_{}".format(idx) in list(self.ops.keys()):
      print("op exists!")
      print(self.graph)
      print(self.ops)
      exit()
    else:
      self.ops["op_{}".format(idx)] = new["op_1"]    
      self.ops["op_{}_dil".format(idx)] = new["op_1_dil"]    
      self.ops["op_{}_kernel".format(idx)] = new["op_1_kernel"]    
      self.ops["op_{}_stride".format(idx)] = new["op_1_stride"]
      print("added {} as operation {} for edge {}".format(new["op_1"],idx,self.graph[-1]))
  def __repr__(self):
    return "Model(Accuracy: {} , Recall: {})".format(self.accuracy,self.recall)

  def set_value(self, name, value):
    self._arch[name] = value
  def get_params(self):
    return list(self.arch().keys())
  def arch(self):
    return self._arch

def train_and_eval_population(worker, population, SETTINGS, train):
  configs = []
  for model in population:
    if model.accuracy == None:
      configs.append({"graph":model.graph,"ops":model.ops})
  acc , _rec , _config = train.eval( configs )
  for mod,result,recall in zip(population, acc, _rec):
    mod.accuracy = result
    mod.recall = recall

def model_change(parent, child):
  p_dict , c_dict = parent.arch().get_dictionary(), child.arch().get_dictionary()
  for _,i in zip(p_dict, c_dict):
    if p_dict[i] != c_dict[i]:
      print("Mutated {} from {} to {}".format(i, p_dict[i], c_dict[i]))
      return True
  print("WARNING: no change found!!!")
  return False



def Mutate(cs, parent_model : Model) -> Model:
  model = copy.deepcopy(parent_model) 
  model.accuracy = None
    
  def op_mutation(model: Model) -> Model:
    operation = random.choice(list(model.ops.keys()))
    model.exchange_one(operation)
    return model
     
  def graph_mutation(model : Model) -> Model:
    old = copy.deepcopy(model.graph)
    model.graph = gen_iter(model.graph)
    if len(old) == len(model.graph):
      print("graph mutate failed")
      print(old)
      print(model.graph)
      exit()
    model.generate_op()
    for i in model.graph:
      if not i in old:
        print("Mutated adding edge {}".format(i))
        return model
    print("No difference in graphs")
    print(old)
    print(model.graph)
    exit()
  
  return random.choice([op_mutation, graph_mutation])(model)    

def regularized_evolution(configspace, worker ,SEARCH_SETTINGS : dict,json_config):
  population = []
  history = []  # Not used by the algorithm, only used to report results.

  CS = configspace
  # Initialize the population with random models.
  train = train_eval( worker , json_config)

  while len(population) < SEARCH_SETTINGS["POPULATION_SIZE"]:
    model = Model( CS ,ops_list = SEARCH_SETTINGS["OPS"])
    model.set_arch( CS.sample_configuration()[0] )
    population.append(model)
    history.append(model)
  train_and_eval_population(worker, population, SEARCH_SETTINGS, train )
  # Carry out evolution in cycles. Each cycle produces a model and removes
  # another.
  children = []
  print_counter = 0
  while len(history) < SEARCH_SETTINGS["TOTAL_EVALUATIONS"]:

    print("Starting cycle", len(history) - SEARCH_SETTINGS["POPULATION_SIZE"])
    # Sample randomly chosen models from the current population.

    if len(children) < SEARCH_SETTINGS["CORES"]:
      sample = []
      while len(sample) < SEARCH_SETTINGS["SAMPLE_SIZE"]:
        # Inefficient, but written this way for clarity. In the case of neural
        # nets, the efficiency of this line is irrelevant because training neural
        # nets is the rate-determining step.
        candidate = random.choice(list(population))
        sample.append(candidate)

      # The parent is the best model in the sample.
      print("Selecting Parent via F1 Score")
      parent = max(sample, key=lambda i: i.accuracy)
      print("Scores of Parent F1: {} -- Acc: {} -- Recall: {}".format(F1(parent.accuracy,parent.recall),parent.accuracy,parent.recall))
      #parent = max(sample, key=lambda i: i.accuracy)

      # Create the child model and store it.
      print("Selected Parent: {}".format(parent.graph))
      child = Mutate(CS,parent)
      print("Mutated child: {}".format(child.graph))
      children.append(child)

    else:
      train_and_eval_population(worker, children, SEARCH_SETTINGS["CORES"], train)
      for i in children:
        population.append(i)
        history.append(i)
        population.pop(0)
      children = []
      # Remove the oldest model.
      best = max(population, key=lambda i: i.accuracy)
      print("--Current best model--")
      print("Accuracy:{} -- Recall: {} ".format(best.accuracy,best.recall))
      print("Architecture: ", best.arch())
      print("Population Size: ", len(population))
      print("Total Evaluations: ", len(history))
      g = nx.DiGraph()
      g.add_edges_from(best.graph)
      #print("nodes in topological sort:{}".format(len([x for x in nx.topological_sort(g)])))
      fixed = {"S":(-10000,0),"T":(20,0)}
      node = fixed.keys()
      pos = nx.spring_layout(g)#,k=0.5, iterations=200
      plt.figure(figsize = (19,12))
      e_labels = {}
      for i,(e,o) in enumerate(zip(best.graph,best.ops)):
        e_labels[e] = "{} (k:{},s:{},d:{})".format(best.ops["op_{}".format(i)],best.ops["op_{}_kernel".format(i)],best.ops["op_{}_stride".format(i)],best.ops["op_{}_dil".format(i)])
      nx.draw(
        g, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,
        labels={node: node for node in g.nodes()}
        )
      nx.draw_networkx_edge_labels(
      g, pos,
      edge_labels=e_labels,
      font_color='red')
      plt.axis('off')
      plt.savefig("{}/best_mode_iter_{}".format(SEARCH_SETTINGS["PATH"],len(history)))
      print_counter +=1
      print_counter = 0
      accuracy_scores = []
      for i in history:
        accuracy_scores.append(i.accuracy)

      plot_scores(accuracy_scores)
  return history


def main(worker, configspace, json_config):
  with open(json_config) as f:
    SETTINGS = json.load(f)["SEARCH_CONFIG"]

  history = regularized_evolution(configspace, worker, SETTINGS,json_config)

if __name__ == '__main__':
  main()
