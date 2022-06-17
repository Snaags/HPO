import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import style
class LivePlot:
  def __init__(self, data_queue):
    style.use('fivethirtyeight')
    self.max = 0
    self.fig = plt.figure()
    self.ax1 = self.fig.add_subplot(1,1,1) 
    self.loss = []
    self.queue = data_queue
    self.ani = animation.FuncAnimation(self.fig, self.animate, interval = 200)

  def fit(self, y):
    x = np.arange(start = 0 ,stop = len(y))
    trend = np.polyfit(x,y,10)
    trendpoly = np.poly1d(trend)
    return x, trendpoly 
    
  def animate(self,i):
      message = self.queue.get(timeout = 30)

      
      self.ax1.clear()
      self.ax1.semilogy(message, lw = 0.5)
      if len(message) > 100:
        x, poly = self.fit(message)
        self.ax1.semilogy(x, poly(x), lw = 1.5, c = "r")
  def decode(self, message):
      pass      
  def show(self):
      plt.show()

def worker_cv( dataset, train_eval ):
    kfold = KFold(n_splits = 20, shuffle = True)
    for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
      train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
      test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
     
      trainloader = torch.utils.data.DataLoader(
                          dataset,collate_fn = collate_fn_padd, 
                          batch_size=batch_size, sampler=train_subsampler, drop_last = True)
      testloader = torch.utils.data.DataLoader(
                          dataset,collate_fn = collate_fn_padd,
                          batch_size=1, sampler=test_subsampler)



