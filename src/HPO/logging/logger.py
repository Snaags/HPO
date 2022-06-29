import pandas as pd
import numpy as np



class log:
  def __init__(self, save_path = "train_log.csv",columns,queue):
    self.db = pd.DataFrame(columns)
    self.queue = queue
    self.save_path = save_path
  def save(self):
    self.db.to_csv(self.save_path)

  def update(self):
    data = self.queue.get()
    self.db.append(data)
    self.save_db()
        
      


