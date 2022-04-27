import torch
import random
from torch.nn.functional import interpolate
import time
import numpy as np
from operator import itemgetter
from scipy.interpolate import CubicSpline
def time_test(func, n = 100, batch_size = 10 , window_length = 1000, features = 27):
    start = time.time()
    x = torch.rand((batch_size, window_length, features))

    for i in range(n):
        out = func(x)
    print("Total time for ",func.__name__,": ", time.time()- start , " Seconds")


def jitter(x : torch.Tensor, y,sigma=0.01, device = None):
    
    n = torch.distributions.normal.Normal(loc=0., scale=sigma)
    return torch.add(x,n.sample(x.shape).cuda(device = device)),y 



def scaling(x : torch.Tensor, y,sigma=0.05, device = None):

    # https://arxiv.org/pdf/1706.00527.pdf
    n = torch.distributions.normal.Normal(loc=1., scale=sigma)
    s = n.sample((x.shape[0],x.shape[2])).cuda(device = device)
    return torch.mul(x, s[:,None,:]),y


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

def crop(x : torch.Tensor,y, crop_min = 0.85, crop_max = 0.95, device = None):
  sig_len = x.shape[2]
  length= random.uniform(crop_min,crop_max)
  length = int(length * sig_len)
  if random.choice([0,1]) == 1:
    return  x[:,:,:length],y
  else:
    return  x[:,:,(sig_len-length):],y

def window_warp(x : torch.Tensor,y, num_warps = 3, ratios = [0.5,0.75,2],size = 0.3, device = None):
    for i in range(num_warps):
        start = random.randint(1, x.shape[2]-10) 
        end = min([x.shape[2],start+random.randint(2, int(x.shape[2]*size))])
        out= interpolate(x[:,:,start:end], scale_factor = random.choice(ratios)).cuda(device = device)
        x = torch.cat((x[:,:,:start],out,x[:,:,end:]),dim = 2)
    if x.shape[2] > 10000:

        x = x[:,:,-10000:]

    return x,y

def cutout(x,y, perc=.1, device = None):    
    seq_len = x.shape[2]    
    win_len = int(perc * seq_len)    
    start = np.random.randint(0, seq_len-win_len-1)    
    end = start + win_len    
    start = max(0, start)    
    end = min(end, seq_len)    
    #print("[INFO] start={}, end={}".format(start, end))    
    x[:,:,start:end] = 0    
    # return new_ts, ts[start:end, ...]    
    return x,y 

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
if __name__ == "__main__":

    from HPO.data.datasets import Test_repsol_full , Mixed_repsol_full
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.data import DataLoader
    import random
    from HPO.utils.worker_helper import train_model, collate_fn_padd
    from HPO.utils.weight_freezing import freeze_all_cells
    import timeit
    import matplotlib.pyplot as plt
    funcs = [cut_mix]
    batch_size = 512
    window_length = 500
    features = 27
    train_dataset = Mixed_repsol_full(0, augmentations_on = False)
    train_dataloader = DataLoader( train_dataset, batch_size=4,
      shuffle = True,drop_last=True, collate_fn = collate_fn_padd)
    for s,l in train_dataloader:
        x = s.cuda()
        y = l.cuda()
        for func in funcs:
            print(x.shape)
            print(y)
            print(func)
            plt.plot(x[0,26,:].cpu(),label = "orig")
            x_p,y_p = func(x,y,device = 0)
            x_p = x_p[0,26,:].cpu()
            plt.plot(x_p ,alpha = 0.5,label = "aug")
            print(y_p)
            plt.legend()
            plt.show()
      #print("Total time for ",func.__name__,": ", timeit.timeit("{}(x)".format(func.__name__), "from __main__ import {}, {}".format(func.__name__, "x"), number = 10) , " Seconds")

