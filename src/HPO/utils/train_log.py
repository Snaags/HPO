import numpy as np 
import pandas as pd
import os

class Logger:
  def __init__(self,path = "train_df", save_interval = 40):
    self.data = {}
    self.save_interval = save_interval
    self.path = path
    while os.path.exists(self.path):
      self.path = self.path + "0"
    self.iter = 0
  def update(self,data_dict):
    self.iter +=1 
    for i in data_dict:
      if i in self.data:
        self.data[i].append(data_dict[i])
      else:
        self.data[i] = [data_dict[i]]
    if self.iter > self.save_interval:
      self.save()
      self.iter = 0
  def save(self):
    df = pd.DataFrame.from_dict(self.data)
    df.to_csv(self.path)
