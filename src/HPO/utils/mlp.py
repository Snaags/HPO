import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_size, layers , output_size):
    super().__init__()
    self.stem = nn.Linear(input_size , layers[0])
    self.hidden = nn.ModuleList()
    self.act = nn.ReLU()
    self.outact = nn.Softmax()
    if len(layers) > 1:
      for c,i in enumerate(layers[1:]):
        self.hidden.append(nn.Linear(layers[c], i))
    self.out = nn.Linear(layers[-1], output_size)

  def forward( self, x ):
    x_s = self.act(self.stem(x))
    if len(self.hidden) > 0:
      for i in self.hidden:
        x_s = self.act(i(x_s))
    x_s = self.act(self.out(x_s))
    return x_s
     
