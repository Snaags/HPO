import random
import torch
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from HPO.utils.model_constructor import Model
from torch.utils.data import Dataset, DataLoader, Sampler

def cal_acc(y,t):
  c = np.count_nonzero(y.T==t)
  tot = len(t)
  return c , tot

def convert_label_max_only(y):
  y = y.cpu().detach().numpy()
  idx = np.argmax(y,axis = 1)
  out = np.zeros_like(y)
  for count, i in enumerate(idx):
      out[count, i] = 1
  return idx

def stdio_print_training_data( iteration : int , outputs : Tensor, labels : Tensor , 
            epoch : int, epochs : int, correct :int , total : int , peak_acc : float , 
            loss : Tensor, n_iter, loss_list = None, binary = True):

  if binary == True:
    new_correct , new_total = cal_acc(outputs.ge(0.5).cpu().detach().numpy(), labels.cpu().detach().numpy())
  else:
    new_correct , new_total =  cal_acc(convert_label_max_only(outputs), convert_label_max_only(labels))
  
  correct += new_correct 
  total += new_total
  acc = correct / total 
  
  if acc > peak_acc:
    peak_acc = acc

  # Save the canvas
  print("Epoch (",str(epoch),"/",str(epochs), ") Accuracy: ","%.2f" % acc, 
              "Iteration(s) (", str(iteration),"/",str(n_iter), ") Loss: ",
              "%.2f" % loss," Correct / Total : {} / {} ".format(correct , total),  end = '\r')

  return correct ,total ,peak_acc



class BalancedBatchSampler(Sampler):
    def __init__(self, dataset,batch_size, indices = None):
        self.dataset = dataset
        if indices is None:
          self.indices = list(range(len(dataset)))
        else:
          self.indices = indices
        self.num_classes = dataset.get_n_classes()
        self.batch_size = batch_size
        # create a dictionary of class labels and their respective indices
        self.label_to_indices = {label: [] for label in range(self.num_classes)}
        for index in self.indices:
            label = dataset[index][1].item()
            self.label_to_indices[label].append(index)

        # ensure that batches are filled completely
        print( self.label_to_indices )
        #assert len(self.indices) % self.num_classes == 0
        self.samples_per_class = self.batch_size // self.num_classes
        self.remainder = self.batch_size % self.num_classes

    def __iter__(self):
        # shuffle labels for each class
        i = 0
        while i < len(self.indices) // self.batch_size:
          indices = []
          for label in self.label_to_indices:
              indices.extend(list(np.random.choice(self.label_to_indices[label],
                size = self.samples_per_class)))
          indices.extend(list(np.random.choice(self.label_to_indices[2],
                size = self.remainder)))    
          np.random.shuffle(indices)
          yield indices
          i+=1 

    def __len__(self):
        return len(self.indices) // self.batch_size


def highest_power_of_two(N):
    power = 1
    while N > 2**power:
      power+=1
    return 2**(power-1)


def get_batch_size_from_n_batches(n_batches : int, samples : int):
    raw_batch_size = samples / n_batches
    if raw_batch_size <= 2:
      return 2
    return 2**int(math.log(raw_batch_size)/math.log(2))



def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t[0].shape[1] for t in batch ])    ## padd
    batch_samples = [ torch.transpose(t[0],0,1) for t in batch ]
    batch_samples = torch.nn.utils.rnn.pad_sequence(batch_samples ,batch_first = True)
    auxlabels = []
    ## compute mask
    mask = (batch != 0)
    labels = []
    for pair in batch:
      one_hot_label = pair[1]
      labels.append(one_hot_label)
      auxlabels.append(pair[2])
    labels = torch.stack(labels).squeeze()
    auxlabels = torch.stack(auxlabels).squeeze()
    #labels = torch.Tensor([t[1] for t in batch])
    batch = torch.transpose(batch_samples , 1 , 2 )
    return batch, labels, auxlabels

def collate_fn_padd_x(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[1] for t in batch ])    ## padd
    batch_samples = [ torch.transpose(t,0,1) for t in batch ]
    batch_samples = torch.nn.utils.rnn.pad_sequence(batch_samples ,batch_first = True)
    ## compute mask
    mask = (batch != 0)
     

    batch = torch.transpose(batch_samples , 1 , 2 )
    return batch


def test_get_batch_size_from_n_batches():
    # Helper function to check if a number is a power of 2
    def is_power_of_2(num):
        return num != 0 and ((num & (num - 1)) == 0)
    
    # Test 1: Basic functionality check
    batch_size = get_batch_size_from_n_batches(4, 32)
    assert batch_size == 8, f"Expected 8 but got {batch_size}"

    # Test 2: Check if returned batch size is a power of 2
    batch_size = get_batch_size_from_n_batches(5, 64)
    assert is_power_of_2(batch_size), f"{batch_size} is not a power of 2"

    # Test 3: The function should divide samples into batches without remainder
    n, s = 3, 32
    batch_size = get_batch_size_from_n_batches(n, s)
    assert s % batch_size == 0, f"{batch_size} does not divide {s} without remainder"
    
    # Test 4: Edge cases - n = 2
    batch_size = get_batch_size_from_n_batches(1, 64)
    assert batch_size == 64, f"Expected 64 but got {batch_size}"

    # Test 5: Edge cases - s = 2
    batch_size = get_batch_size_from_n_batches(5, 1)
    assert batch_size == 2, f"Expected 2 but got {batch_size}"

    # Test 6: Edge cases - n = s
    batch_size = get_batch_size_from_n_batches(64, 64)
    assert batch_size == 2, f"Expected 2 but got {batch_size}"

    print("All tests passed!")

if __name__ == "__main__":
  test_get_batch_size_from_n_batches()