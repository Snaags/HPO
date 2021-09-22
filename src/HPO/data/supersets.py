import random


class SuperSet:
  def __init__(self, dataloaders):
    self.dataloaders = dataloaders
    self.iterators = []
    self.set_iterator()
  def __getitem__(self, index):
    rand_index = self.get_idx()
    try:
      return next(self.iterators[rand_index])
    except StopIteration:
      self.iterators.pop(rand_index)
    if len(self.iterators) == 0:
      raise StopIteration
    rand_index = self.get_idx()
    return next(self.iterators[rand_index])

  def set_iterator(self):
    for i in self.dataloaders:
      self.iterators.append(iter(i))

  def get_idx(self):
    if len(self.iterators) == 1:
      return 0
    elif len(self.iterators) == 0:
      self.set_iterator()
      raise StopIteration


    else:
      return random.randint(0,len(self.iterators) -1)

  def __len__(self):
    return sum([len(x) for x in self.dataloaders])  










