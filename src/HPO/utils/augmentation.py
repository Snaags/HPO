import matplotlib.pyplot as plt
import torch
import random
from torch.nn.functional import interpolate
import time
import math
import numpy as np
from operator import itemgetter
from scipy.interpolate import CubicSpline


def initialise_augmentations(augmentation_data,device = None)->list:
  if augmentation_data:
    augs = []
    for i in augmentation_data:
      augs.append(eval(i.split("_")[0])(**augmentation_data[i],device = device) )
    return augs
  else:
    return False

class Augmentation(object):
  def __init__(self,rate,device):
    self.device = device
    self.rate = rate
  def set_train_ids(self,train_ids,dataset):
    self.ids = train_ids
    self.dataset = dataset
  def __call__(self,x,y):
    rate = self.rate
    while random.random() < rate:
      x,y = self.call(x,y)
      rate -= 1
    else:
      return x,y
  def call(self,x,y):
    print("called parent class function :(")

class Jitter(Augmentation):
  def __init__(self,sigma = 0.05,rate = 0.3,device = None):
    super(Jitter,self).__init__(rate,device)
    self.n = torch.distributions.normal.Normal(loc=torch.FloatTensor([0]).cuda(device =device), scale=torch.FloatTensor([sigma]).cuda(device = device))
    self.__name__ = "jitter"
  def call(self,x,y):
    return torch.add(x,self.n.sample(x.shape).squeeze()),y 

class Scaling(Augmentation):
  def __init__(self,sigma = 0.05,rate = 0.3 ,device = None):
    super(Scaling,self).__init__(rate,device)
    self.n = torch.distributions.normal.Normal(loc=torch.FloatTensor([1]).cuda(device =device), scale=torch.FloatTensor([sigma]).cuda(device = device))
    self.__name__ = "scaling"
  def call(self,x,y):
    s = self.n.sample((x.shape[0],x.shape[1])).squeeze()
    return torch.mul(x, s),y

class Crop(Augmentation):
  def __init__(self, crop_min = 0.5,rate = 0.3, crop_max = 0.99, device = None):
    super(Crop,self).__init__(rate,device)
    self.__name__ = "crop"
    self.crop_min = crop_min
    self.crop_max = crop_max
  def call(self,x,y):
    sig_len = x.shape[1]
    length= random.uniform(self.crop_min,self.crop_max)
    length = int(length * sig_len)
    start = random.randint(0,(sig_len - length))
    return x[:,start:(length+start)],y

    if random.choice([0,1]) == 1:
      return  x[:,:length],y
    else:
      return  x[:,(sig_len-length):],y

class WindowWarp(Augmentation):
  def __init__(self, num_warps = 3, ratios = [0.5,0.75,2],size = 0.3,rate = 0.3, device = None):
    super(WindowWarp,self).__init__(rate,device)
    self.__name__ = "window_warp"
    self.ratios = ratios
    self.num_warps = num_warps
    self.size = size
  def call(self,x,y):
    for i in range(self.num_warps):
        start = random.randint(1, x.shape[1]-3) 
        end = min([x.shape[1],start+random.randint(2, math.ceil(x.shape[1]*self.size))])
        out= interpolate(x[:,start:end].unsqueeze(0), scale_factor = random.choice(self.ratios)).cuda(device = self.device).squeeze(0)
        x = torch.cat((x[:,:start],out,x[:,end:]),dim = 1)
    if x.shape[1] > 20000:
        print("Window Warp exceeded size: {}".format(x.shape[1]))
    return x,y
class CutOut(Augmentation):
  def __init__(self,perc=.1,rate = 0.3,device = None):
    super(CutOut,self).__init__(rate,device)
    self.perc = perc
    self.__name__ = "cut_out"
  def call(self,x,y):
    seq_len = x.shape[1]    
    win_len = int(self.perc * seq_len)
    if seq_len-win_len-1 < 1:
      return x,y
    start = np.random.randint(0, seq_len-win_len-1)    
    end = start + win_len    
    start = max(0, start)    
    end = min(end, seq_len)    
    x[:,start:end] = 0    
    return x,y 

class ChannelDrop(Augmentation):
  def __init__(self, crop_min = 0.6,rate = 0.3, crop_max = 0.99, device = None):
    super(ChannelDrop,self).__init__(rate,device)
    self.__name__ = "crop"
    self.crop_min = crop_min
    self.crop_max = crop_max
  def call(self,x,y):
    n_channels = x.shape[0]
    p_drop = random.uniform(self.crop_min,self.crop_max)
    n_drop = int(n_channels * p_drop)
    x[np.random.choice(np.arange(n_channels),n_drop),:] = 0
    return x,y


class MixUp(Augmentation):
  def __init__(self, m = 0.3 , rate = 0.3 , device = None):
    super(MixUp,self).__init__(rate,device)
    self.dist = torch.distributions.beta.Beta(m,m)
    self.mix_sample = None
    self.mix_label = None 
  def __call__(self,x,y):
    if random.random() < self.rate:
      self.mix_sample,self.mix_label = self.dataset.x[random.choice(self.ids)],self.dataset.y[random.choice(self.ids)]
      return self.call(x,y)
    else:
      return x,y



  def call(self,x,y):
    DEBUG = False
    #x = [batch, channels , length]
    #y = [batch, 1]
    mix = self.dist.sample()
    MTS_1 = x.clone()
    MTS_2 = self.mix_sample
    LAB_1 = y.clone()
    LAB_2 = self.mix_label
    x = (MTS_1 * mix)  + (MTS_2 *(1-mix))
    y = (LAB_1 * mix) + (LAB_2 * (1-mix))
    if DEBUG == True:
      print("mix value is {} and label value is {}".format(mix,y))
      plt.plot(MTS_1[26,:].cpu(),label = "series 1",alpha = 0.7)
      plt.plot(MTS_2[26,:].cpu(),label = "series 2",alpha = 0.7)
      plt.plot(x[26,:].cpu(),label = "Mix",alpha = 0.7)
      plt.legend()
      plt.show()
    self.mix_sample = MTS_1
    self.mix_label = LAB_1
    return x ,y.long()



class ChannelCutMix(Augmentation):
    def __init__(self, perc=0.1, rate=0.3, device=None):
        super(ChannelCutMix, self).__init__(rate,device=device)  # Assuming device is used in the parent class
        self.perc = perc
        self.rate = rate
        self.mix_sample = None
        self.mix_label = None

    def __call__(self, x, y):
        if random.random() < self.rate:
            self.mix_sample, self.mix_label = self.dataset.x[random.choice(self.ids)], self.dataset.y[random.choice(self.ids)]
            return self.call(x, y)
        else:
            return x, y

    def call(self, x, y):
        n_channels = x.shape[0]
        n_drop = int(n_channels * self.perc)
        channels_to_mix = np.random.choice(np.arange(n_channels),n_drop, replace=False)
        x[channels_to_mix,:] = self.mix_sample[channels_to_mix,:]

        # This assumes y is a one-hot encoded vector, and thus can be averaged
        y = (y * (1-self.perc)) + (self.mix_label * self.perc)
        return x, y.long()



class CutMix(Augmentation):
    def __init__(self, perc=0.1, rate=0.3, device=None):
        super(CutMix, self).__init__(rate,device=device)  # Assuming device is used in the parent class
        self.perc = perc
        self.rate = rate
        self.mix_sample = None
        self.mix_label = None

    def __call__(self, x, y):
        if random.random() < self.rate:
            self.mix_sample, self.mix_label = self.dataset.x[random.choice(self.ids)], self.dataset.y[random.choice(self.ids)]
            return self.call(x, y)
        else:
            return x, y

    def call(self, x, y):
        seq_len = x.shape[1]
        win_len = int(self.perc * seq_len)
        start = np.random.randint(0, seq_len - win_len - 1)
        end = start + win_len

        # This assumes the data is in the shape [channels, length]
        x[:, start:end] = self.mix_sample[:, start:end]
        # This assumes y is a one-hot encoded vector, and thus can be averaged
        y = (y * (1-self.perc)) + (self.mix_label * self.perc)
        return x, y.long()







def mix_up(x,y, m = 0.3, device = None):
    dist = torch.distributions.beta.Beta(m,m)
    DEBUG = False
    #x = [batch, channels , length]
    #y = [batch, 1]
    for i in range(x.shape[0]):
      i_2 = list(range(x.shape[0]))
      i_2.remove(i)
      i_2 = random.choice(i_2)
      mix = dist.sample()
      MTS_1 = x[i,:,:].clone()
      MTS_2 = x[i_2,:,:].clone()
      LAB_1 = y[i]
      LAB_2 = y[i_2]
      x[i,:,:] = (MTS_1 * mix)  + (MTS_2 *(1-mix))
      y[i] = (LAB_1 * mix) + (LAB_2 * (1-mix))
      if DEBUG == True:
        print("mix value is {} and label value is {}".format(mix,y[i]))
        plt.plot(MTS_1[26,:].cpu(),label = "series 1",alpha = 0.7)
        plt.plot(MTS_2[26,:].cpu(),label = "series 2",alpha = 0.7)
        plt.plot(x[i,26,:].cpu(),label = "Mix",alpha = 0.7)
        plt.legend()
        plt.show()
    return x ,y

def cut_mix(x,y, perc=.1, device = None):    
    seq_len = x.shape[2]    
    win_len = int(perc * seq_len)    
    start = np.random.randint(0, seq_len-win_len-1)    
    end = start + win_len    
    for i in range(x.shape[0]):    
        start = max(0, start)    
        end = min(end, seq_len)    
        x[i,:,start:end] = x[random.randint(0,x.shape[0]-1),:,start:end]    
    # return new_ts, ts[start:end, ...]    
    return x,y 






















def rotation(x : torch.Tensor,y):
    flip = torch.randint(0,2,size = (x.shape[0],x.shape[2]))
    flip = torch.where(flip != 0, flip, -1)
    rotate_axis = torch.arange(x.shape[2])
    s = np.random.shuffle(np.array(range(x.shape[2])))
    rotate_axis[np.array(range(x.shape[2]))] = rotate_axis[s].clone()
    return flip[:,None,:] * x[:,:,rotate_axis],y



def permutation(x : torch.Tensor,y, max_segments=5, seg_mode="equal"):
    #make array of ints from 1 to window_length
    num_seqments_per_batch = torch.randint(0 , max_segments, 
        size = (x.shape[0],))
    #loop through batches and split into the 
    #defined number of segments for each 
    for idx, sequence in enumerate(x):
        if num_seqments_per_batch[idx] > 1:

            #returns a list of tensors
            chunk_list = torch.chunk(sequence, num_seqments_per_batch[idx], dim = 0)
            #Shuffle
            shuffled_chunks = itemgetter(*np.random.permutation(len(chunk_list)))(chunk_list)
            permutated_sequence = torch.cat(shuffled_chunks, dim = 0)
            x[idx] = permutated_sequence


    return x,y






def magnitude_warp(x : torch.Tensor,y, sigma=0.2, knot=4):
    def h_poly(t):
        tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=t.dtype, device=t.device)
        return A @ tt


    def interp(x, y, xs):
        m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
        idxs = torch.searchsorted(x[1:], xs)
        dx = (x[idxs + 1] - x[idxs])
        hh = h_poly((xs - x[idxs]) / dx)
        return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx

    orig_steps= torch.linspace(0, x.shape[1]-1, x.shape[1])
    n = torch.distributions.normal.Normal(loc=1., scale=sigma)
    random_warps = n.sample((x.shape[0],knot+2 , x.shape[2]))
    warp_steps = (torch.ones((x.shape[2],1))*(torch.linspace(0, x.shape[1]-1., knot+2))).T
    ret = torch.zeros(x.shape)
    warper = torch.zeros(orig_steps.shape[0],x.shape[2])
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            warper[:,dim] =  interp(warp_steps[:,dim], random_warps[i,:,dim],  orig_steps)

        ret[i] = pat * warper

    return ret,y

if __name__ == "__main__":

    import torch.nn as nn
    from torch import Tensor
    from HPO.data.teps_datasets import Train_TEPS
    from torch.utils.data import DataLoader
    import random
    import timeit
    import matplotlib.pyplot as plt
    jitter = Jitter(device = 0)
    scaling = Scaling(device = 0)
    window_warp = WindowWarp(device = 0)
    crop = Crop(device = 0)
    cut_out = CutOut(device = 0)
    funcs = [crop,jitter,scaling,window_warp,cut_out]
    batch_size = 1
    window_length = 500
    features = 27
    train_dataset = Train_TEPS()
    x,y = next(iter(train_dataset))
    print(x.shape)
    train_dataloader = DataLoader( train_dataset, batch_size=batch_size,
      shuffle = True,drop_last=True)
    for i,(s,l) in enumerate(train_dataloader):
        x = s.cuda().squeeze()
        y = l.cuda()
        print("Shape of sample: {}".format(x.shape))
        if i > 0:
          break
        for func in funcs:
            """
            print(x.shape)
            print(func)
            #plt.plot(x[0,26,:].cpu(),label = "orig")
            x_p,y_p = func(x,y,device = 0)
            x_p = x_p[0,26,:].cpu()
            #plt.plot(x_p ,alpha = 0.5,label = "aug")
            #plt.legend()
            #plt.show()
            """
            print("from __main__ import {}, {}, {}".format(func.__name__, "x","y"))
            t = timeit.timeit("{}(x,y)".format(func.__name__), "from __main__ import {}, {}, {}".format(func.__name__, "x","y"), number = 1000)
            print("Total time for ",func.__name__,": ",t , " Seconds")
            print("Time per sample: {}".format(t/batch_size))

