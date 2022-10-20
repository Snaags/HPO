import torch
import torch.nn as nn




# input - > training loss , training accuracy, validation accuracy, validation loss

#output -> augmentation params , lr 

 
class AutoTrainer(nn.Module):
  def __init__(self,channels):
    super(AutoTrainer,self).__init__()

    self.channels = channels
    self.stem = nn.Conv1d(2 , channels[0] , 3)
    self.GAP = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(self.channels, 1)
    self.act = nn.ReLU()
    self.fcact = nn.Softmax()
    optimizer = torch.optim.Adam(self.parameters(),lr = 0.0025)
 
  def forward(self, x):
    x = self.stem(x)
    x = self.GAP(x)
    x = torch.flatten(x, 1)
    return self.fcact(self.fc(x))


  def update(self,loss):
    
