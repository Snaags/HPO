import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
ACTIVATION = {
  "gelu" : lambda C, L: nn.GELU(),
  "relu" : lambda  C, L: nn.ReLU(),
  "none" : lambda  C, L: Identity(),
  "tanh" : lambda  C, L: nn.Tanh(),
  "mha" : lambda  C, L: MHA(C)
}
NORMALISATION = {
  "batch_norm" : lambda C, l : nn.BatchNorm1d(C),  
  "layer_norm" : lambda C, l : nn.LayerNorm([C,l]),
  "none" : lambda C, l : Identity()

}

COMBINE = {
  "CONCAT" : lambda : CONCAT(),
  "ADD" : lambda : ADD()
}

OPS = {
  "none" : lambda C : Zero(),
  "gelu" : lambda C,kernel, stride,dil, affine: nn.GELU(),
  "relu" : lambda : nn.ReLU(),
  "batch_norm" : lambda C : nn.BatchNorm1d(C),
  "skip_connect" : lambda C : Identity(),
  "MHA_2" : lambda C : MHA(C, 2),
  "MHA_4" : lambda C : MHA(C, 4),
  "MHA_8" : lambda C : MHA(C, 8),
  "SE_1" : lambda C : SE(C, 1),
  "SE_2" : lambda C : SE(C, 2),
  "SE_4" : lambda C : SE(C, 4),
  "lstm" :lambda C : LSTM(C, False),
  "lstm_bi" :lambda C : LSTM(C, True),
  "depth_lstm" :lambda C : DepthLSTM(C, False),
  "depth_lstm_bi" :lambda C : DepthLSTM(C, True),
  "resample_channels" : lambda c_in,c_out: ResampleChannels(c_in,c_out),
  "downsample_resolution" : lambda c_in,stride: DownsampleResolution(c_in,stride),
  "point_conv_1" :  lambda C: PointConv(C, stride = 1,dilation = 1,padding = 1*(3 - 1)//2, affine = True),
  #DEPTHWISE CONVOLUTIONS
  "depth_conv_3_1" :  lambda C: DepthConv(C,3, stride = 1,dilation = 1,padding = 1*(3 - 1)//2, affine = True),
  "depth_conv_5_1" :  lambda C: DepthConv(C,5, stride = 1,dilation = 1,padding = 1*(5 - 1)//2, affine = True),
  "depth_conv_7_1" :  lambda C: DepthConv(C,7, stride = 1,dilation = 1,padding = 1*(7 - 1)//2, affine = True),

  "depth_conv_3_2" :  lambda C: DepthConv(C,3, stride = 1,dilation = 2,padding = 2*(3 - 1)//2, affine = True),
  "depth_conv_5_2" :  lambda C: DepthConv(C,5, stride = 1,dilation = 2,padding = 2*(5 - 1)//2, affine = True),
  "depth_conv_7_2" :  lambda C: DepthConv(C,7, stride = 1,dilation = 2,padding = 2*(7 - 1)//2, affine = True),
  "depth_conv_3_4" :  lambda C: DepthConv(C,3, stride = 1,dilation = 4,padding = 4*(3 - 1)//2, affine = True),
  "depth_conv_5_4" :  lambda C: DepthConv(C,5, stride = 1,dilation = 4,padding = 4*(5 - 1)//2, affine = True),
  "depth_conv_7_4" :  lambda C: DepthConv(C,7, stride = 1,dilation = 4,padding = 4*(7 - 1)//2, affine = True),
  "depth_conv_3_8" :  lambda C: DepthConv(C,3, stride = 1,dilation = 8,padding = 8*(3 - 1)//2, affine = True),
  "depth_conv_5_8" :  lambda C: DepthConv(C,5, stride = 1,dilation = 8,padding = 8*(5 - 1)//2, affine = True),
  "depth_conv_7_8" :  lambda C: DepthConv(C,7, stride = 1,dilation = 8,padding = 8*(7 - 1)//2, affine = True),
  "depth_conv_3_16" :  lambda C: DepthConv(C,3, stride = 1,dilation = 16,padding = 16*(3 - 1)//2, affine = True),
  "depth_conv_5_16" :  lambda C: DepthConv(C,5, stride = 1,dilation = 16,padding = 16*(5 - 1)//2, affine = True),
  "depth_conv_7_16" :  lambda C: DepthConv(C,7, stride = 1,dilation = 16,padding = 16*(7 - 1)//2, affine = True),
  "depth_conv_3_32" :  lambda C: DepthConv(C,3, stride = 1,dilation = 32,padding = 32*(3 - 1)//2, affine = True),
  "depth_conv_5_32" :  lambda C: DepthConv(C,5, stride = 1,dilation = 32,padding = 32*(5 - 1)//2, affine = True),
  "depth_conv_7_32" :  lambda C: DepthConv(C,7, stride = 1,dilation = 32,padding = 32*(7 - 1)//2, affine = True),
  "depth_conv_15_1" :  lambda C: DepthConv(C,15, stride = 1,dilation = 1,padding = 1*(15 - 1)//2, affine = True),
  "depth_conv_31_1" :  lambda C: DepthConv(C,31, stride = 1,dilation = 1,padding = 1*(31 - 1)//2, affine = True),
  "depth_conv_63_1" :  lambda C: DepthConv(C,63, stride = 1,dilation = 1,padding = 1*(63 - 1)//2, affine = True),
  "depth_conv_15_2" :  lambda C: DepthConv(C,15, stride = 1,dilation = 2,padding = 2*(15 - 1)//2, affine = True),
  "depth_conv_31_2" :  lambda C: DepthConv(C,31, stride = 1,dilation = 2,padding = 2*(31 - 1)//2, affine = True),
  "depth_conv_63_2" :  lambda C: DepthConv(C,63, stride = 1,dilation = 2,padding = 2*(63 - 1)//2, affine = True),
  "depth_conv_15_4" :  lambda C: DepthConv(C,15, stride = 1,dilation = 4,padding = 4*(15 - 1)//2, affine = True),
  "depth_conv_31_4" :  lambda C: DepthConv(C,31, stride = 1,dilation = 4,padding = 4*(31 - 1)//2, affine = True),
  "depth_conv_63_4" :  lambda C: DepthConv(C,63, stride = 1,dilation = 4,padding = 4*(63 - 1)//2, affine = True),
  "depth_conv_15_8" :  lambda C: DepthConv(C,15, stride = 1,dilation = 8,padding = 8*(15 - 1)//2, affine = True),
  "depth_conv_31_8" :  lambda C: DepthConv(C,31, stride = 1,dilation = 8,padding = 8*(31 - 1)//2, affine = True),
  "depth_conv_63_8" :  lambda C: DepthConv(C,63, stride = 1,dilation = 8,padding = 8*(63 - 1)//2, affine = True),
  "depth_conv_15_16" :  lambda C: DepthConv(C,15, stride = 1,dilation = 16,padding =16*(15 - 1)//2, affine = True),
  "depth_conv_31_16" :  lambda C: DepthConv(C,31, stride = 1,dilation = 16,padding = 8*(31 - 1)//2, affine = True),
  "depth_conv_63_16" :  lambda C: DepthConv(C,63, stride = 1,dilation = 16,padding = 8*(31 - 1)//2, affine = True),
  "depth_conv_15_32" :  lambda C: DepthConv(C,15, stride = 1,dilation = 32,padding =16*(15 - 1)//2, affine = True),
  "depth_conv_31_32" :  lambda C: DepthConv(C,31, stride = 1,dilation = 32,padding =16*(15 - 1)//2, affine = True),
  "depth_conv_63_32" :  lambda C: DepthConv(C,63, stride = 1,dilation = 32,padding =32*(63 - 1)//2, affine = True),
  #AVERAGE POOLING OPERATIONS
  "avg_pool_3_1" :  lambda C: nn.AvgPool1d(3,stride = 1, padding = 1*(3 - 1)//2,count_include_pad=False),
  "avg_pool_5_1" :  lambda C: nn.AvgPool1d(5,stride = 1,padding = 2,count_include_pad=False),
  "avg_pool_7_1" :  lambda C: nn.AvgPool1d(7,stride = 1,padding = 3,count_include_pad=False),
  "avg_pool_15_1" :  lambda C: nn.AvgPool1d(15,stride = 1,padding = 7,count_include_pad=False),
  "avg_pool_31_1" :  lambda C: nn.AvgPool1d(31,stride = 1,padding = 15,count_include_pad=False),
  "avg_pool_63_1" :  lambda C: nn.AvgPool1d(63,stride = 1,padding = 31,count_include_pad=False),
  #MAXPOOLING OPERATIONS
  "max_pool_3_1" :  lambda C: nn.MaxPool1d(3,stride = 1 , dilation = 1,padding = 1),
  "max_pool_5_1" :  lambda C: nn.MaxPool1d(5,stride = 1 , dilation = 1,padding = 2),
  "max_pool_7_1" :  lambda C: nn.MaxPool1d(7,stride = 1 , dilation = 1,padding = 3),
  "max_pool_15_1" :  lambda C: nn.MaxPool1d(15,stride = 1 , dilation = 1,padding = 7),
  "max_pool_31_1" :  lambda C: nn.MaxPool1d(31,stride = 1 , dilation = 1,padding = 15),
  "max_pool_63_1" :  lambda C: nn.MaxPool1d(63,stride = 1 , dilation = 1,padding = 31),
  "max_pool_3_2" :  lambda C: nn.MaxPool1d(3,stride = 1 , dilation = 2,padding = 2*(3 - 1)//2),
  "max_pool_5_2" :  lambda C: nn.MaxPool1d(5,stride = 1 , dilation = 2,padding = 2*(5 - 1)//2),
  "max_pool_7_2" :  lambda C: nn.MaxPool1d(7,stride = 1 , dilation = 2,padding = 2*(7 - 1)//2),
  "max_pool_3_4" :  lambda C: nn.MaxPool1d(3,stride = 1 , dilation = 4,padding = 4*(3 - 1)//2),
  "max_pool_5_4" :  lambda C: nn.MaxPool1d(5,stride = 1 , dilation = 4,padding = 4*(5 - 1)//2),
  "max_pool_7_4" :  lambda C: nn.MaxPool1d(7,stride = 1 , dilation = 4,padding = 4*(7 - 1)//2),
  "max_pool_3_8" :  lambda C: nn.MaxPool1d(3,stride = 1 , dilation = 8,padding = 8*(3 - 1)//2),
  "max_pool_5_8" :  lambda C: nn.MaxPool1d(5,stride = 1 , dilation = 8,padding = 8*(5 - 1)//2),
  "max_pool_7_8" :  lambda C: nn.MaxPool1d(7,stride = 1 , dilation = 8,padding = 8*(7 - 1)//2),
  "max_pool_3_16" :  lambda C: nn.MaxPool1d(3,stride = 1 , dilation = 16,padding = 16*(3 - 1)//2),
  "max_pool_5_16" :  lambda C: nn.MaxPool1d(5,stride = 1 , dilation = 16,padding = 16*(5 - 1)//2),
  "max_pool_7_16" :  lambda C: nn.MaxPool1d(7,stride = 1 , dilation = 16,padding = 16*(7 - 1)//2),
  "max_pool_3_32" :  lambda C: nn.MaxPool1d(3,stride = 1 , dilation = 32,padding = 32*(3 - 1)//2),
  "max_pool_5_32" :  lambda C: nn.MaxPool1d(5,stride = 1 , dilation = 32,padding = 32*(5 - 1)//2),
  "max_pool_7_32" :  lambda C: nn.MaxPool1d(7,stride = 1 , dilation = 32,padding = 32*(7 - 1)//2),
  "max_pool_15_2" :  lambda C: nn.MaxPool1d(15,stride = 1 , dilation = 2,padding = 2*(15 - 1)//2),
  "max_pool_31_2" :  lambda C: nn.MaxPool1d(31,stride = 1 , dilation = 2,padding = 2*(31 - 1)//2),
  "max_pool_63_2" :  lambda C: nn.MaxPool1d(63,stride = 1 , dilation = 2,padding = 2*(63 - 1)//2),
  "max_pool_15_4" :  lambda C: nn.MaxPool1d(15,stride = 1 , dilation = 4,padding = 4*(15 - 1)//2),
  "max_pool_31_4" :  lambda C: nn.MaxPool1d(31,stride = 1 , dilation = 4,padding = 4*(31 - 1)//2),
  "max_pool_63_4" :  lambda C: nn.MaxPool1d(63,stride = 1 , dilation = 4,padding = 4*(63 - 1)//2),
  "max_pool_15_8" :  lambda C: nn.MaxPool1d(15,stride = 1 , dilation = 8,padding = 8*(15 - 1)//2),
  "max_pool_31_8" :  lambda C: nn.MaxPool1d(31,stride = 1 , dilation = 8,padding = 8*(31 - 1)//2),
  "max_pool_63_8" :  lambda C: nn.MaxPool1d(63,stride = 1 , dilation = 8,padding = 8*(63 - 1)//2),
  "max_pool_15_16" :  lambda C: nn.MaxPool1d(15,stride = 1 , dilation = 16,padding =16*(15 - 1)//2),
  "max_pool_31_16" :  lambda C: nn.MaxPool1d(31,stride = 1 , dilation = 16,padding =16*(31 - 1)//2),
  "max_pool_63_16" :  lambda C: nn.MaxPool1d(63,stride = 1 , dilation = 16,padding =16*(63 - 1)//2),
  "max_pool_15_32" :  lambda C: nn.MaxPool1d(15,stride = 1 , dilation = 32,padding =32*(15 - 1)//2),
  "max_pool_31_32" :  lambda C: nn.MaxPool1d(31,stride = 1 , dilation = 32,padding =32*(31 - 1)//2),
  "max_pool_63_32" :  lambda C: nn.MaxPool1d(63,stride = 1 , dilation = 32,padding =32*(63 - 1)//2),

  'sep_conv_3_BR' : lambda C : SepConv(C, C, 3, 1, 1, affine=1),
  'sep_conv_5_BR' : lambda C : SepConv(C, C, 5, 1, 2, affine=1),
  'sep_conv_7_BR' : lambda C : SepConv(C, C, 7, 1, 3, affine=1),
  'sep_conv_15_BR' : lambda C, : SepConv(C, C, 15, 1, 7, affine=1),
  'sep_conv_30_BR' : lambda C: SepConv(C, C, 27, 1, 13, affine=1),
  'dil_conv_3_BR' : lambda C: DilConv(C, C, 3, 1, 2, 2, affine=1),
  'dil_conv_5_BR' : lambda C: DilConv(C, C, 5, 1, 4, 2, affine=1),
  'dil_conv_7_BR' : lambda C: DilConv(C, C, 3, 1, 2, 2, affine=1),

  "depth_conv_3_1_BR" :  lambda C: DepthConvBR(C,3, stride = 1,dilation = 1,padding = 1*(3 - 1)//2, affine = True),
  "depth_conv_5_1_BR" :  lambda C: DepthConvBR(C,5, stride = 1,dilation = 1,padding = 1*(5 - 1)//2, affine = True),
  "depth_conv_7_1_BR" :  lambda C: DepthConvBR(C,7, stride = 1,dilation = 1,padding = 1*(7 - 1)//2, affine = True),
  "depth_conv_3_2_BR" :  lambda C: DepthConvBR(C,3, stride = 1,dilation = 2,padding = 2*(3 - 1)//2, affine = True),
  "depth_conv_5_2_BR" :  lambda C: DepthConvBR(C,5, stride = 1,dilation = 2,padding = 2*(5 - 1)//2, affine = True),
  "depth_conv_7_2_BR" :  lambda C: DepthConvBR(C,7, stride = 1,dilation = 2,padding = 2*(7 - 1)//2, affine = True),
  "point_conv_1_BR" :  lambda C: PointConvBR(C, stride = 1,dilation = 1,padding = 1*(3 - 1)//2, affine = True),
  
  }

"""
class MHA(nn.Module):
  def __init__(self,C):
    super(MHA,self).__init__()
    self.act = nn.MultiheadAttention(C, C, batch_first = True)
  def forward(self,x):
    x = x.swapaxes(1,2)
    return self.act(x,x,x,need_weights=False)[0].swapaxes(1,2)
"""

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class CONCAT(nn.Module):

  def __init__(self):
    super(CONCAT, self).__init__()

  def forward(self, x):
    return torch.cat(x,dim = 1)

class ADD(nn.Module):

  def __init__(self):
    super(ADD, self).__init__()

  def forward(self, x):
    x = torch.stack(x, dim=0)
    return torch.sum(x,dim = 0)

class Zero(nn.Module):

  def __init__(self):
    super(Zero, self).__init__()

  def forward(self, x):
    return x.mul(0.)


class GRU(nn.Module):
  def __init__(self, c):
    super(GRU, self).__init__()
    self.gru = nn.GRU(
        input_size=c,
        hidden_size=c,
        num_layers=1,
        batch_first=True,
        bias = False
        )

class LSTM(nn.Module):
  def __init__(self, c,bidirectional = False):
    super(LSTM, self).__init__()
    self.lstm = nn.LSTM(
        input_size=c,
        hidden_size=c,
        num_layers=1,
        batch_first=True,
        bias = False
        )
  def forward(self,x):
    x = x.permute(0,2,1)
    x, (_1,_2) = self.lstm(x)
    x = x.permute(0,2,1)
    return x

class DepthLSTM(nn.Module):
  def __init__(self, c,bidirectional = False):
    super(DepthLSTM, self).__init__()
    self.lstms = nn.ModuleList(
      [
            nn.LSTM(input_size=1, hidden_size=1, num_layers=1, batch_first=True,bias = False,bidirectional = bidirectional)
            for _ in range(c)
        ])

  def forward(self,x):
    outputs = []
    for i, lstm in enumerate(self.lstms):
        input_channel = x[:, i, :].unsqueeze(-1)  # Select the ith feature and add feature dimension
        output, (_1,_2) = lstm(input_channel)
        outputs.append(output)
    outputs = torch.cat(outputs, dim=-1)  # Concatenate outputs along the feature dimension
    return outputs.permute(0,2,1)
class DownsampleResolution(nn.Module):
  """
  This operation uses a depthwise convolution to reduce the signal resolution when needed.
  Given a stride of 2 its a kernel of 2 etc. 
  """
  def __init__(self, c_in,stride):
    super(DownsampleResolution,self).__init__()
    if stride > 2:
      print("Stride: {}".format(stride))
    self.conv1 = nn.Conv1d(c_in,c_in, kernel_size = stride, stride = stride, bias = False, groups = c_in)
  def forward(self,x):
    return self.conv1(x)

class ResampleChannels(nn.Module):
  """
  This operation uses a pointwise convolution to change the number of channels when needed.
  """
  def __init__(self,c_in,c_out):
    super(ResampleChannels, self).__init__()
    self.conv1 = nn.Conv1d(c_in,c_out, kernel_size = 1, stride = 1, bias = False)
  def forward(self,x):
    return self.conv1(x)

class GELU(nn.Module):
  def __init__(self):
    super(GELU,self).__init__()
    self.act = nn.GELU()
  def forward(self,x):
    return self.act(x)

class SEMIX(nn.Module):
  def __init__(self, C_in,C_out,r =2 ,stride =1,affine = True ):
    super(SEMIX,self).__init__()
    #print("Building Squeeze Excite with input {} and output: {}".format(C_in,C_out))
    self.GP = nn.AdaptiveAvgPool1d(1)
    self.fc1 = nn.Linear(C_in, C_in//2, bias = False)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(C_in//2, C_out ,bias = False)
    self.sig = nn.Sigmoid()
    self.stride = stride
  def forward(self,x1,x2):
    #Squeeze
    y = self.GP(x2).squeeze()# [Batch,C]
    #torch.mean(x,axis = 2)  
    y = self.fc1(y)
    y = self.act(y)
    y = self.sig(y).unsqueeze(dim = 2)
    return x1* y.expand_as(x1)


class LayerNorm(nn.Module):
  def __init__(self,c,affine):
    super(LayerNorm,self).__init__()
    self.norm = nn.LayerNorm(c, affine=affine)
  def forward(self,x):
    return self.norm(x)

class BatchNorm(nn.Module):
  def __init__(self,c,affine):
    super(BatchNorm,self).__init__()
    self.norm = nn.BatchNorm1d(c, affine=affine)
  def forward(self,x):
    return self.norm(x)

class DepthConv(nn.Module):
  def __init__(self, c, kernel_size, padding,stride,dilation= 1, affine=True):
    super(DepthConv, self).__init__()
    #padding = (kernel_size*stride*dilation)//2
    self.conv = nn.Conv1d(c,c, kernel_size=kernel_size, stride=stride, padding="same", groups=c, bias=False)

  def forward(self, x):
    x =self.conv(x)
    return x

class DepthConvBR(nn.Module):
  def __init__(self, c, kernel_size, padding,stride,dilation= 1, affine=True):
    super(DepthConvBR, self).__init__()
    #padding = (kernel_size*stride*dilation)//2
    self.act =  nn.ReLU()
    self.bn = nn.BatchNorm1d(c, affine=affine)
    self.conv = nn.Conv1d(c,c, kernel_size=kernel_size, stride=stride, padding="same", groups=c, bias=False)

  def forward(self, x):
    x = self.act(x)
    x =self.conv(x)
    x = self.bn(x)
    return x

class PointConv(nn.Module):
  def __init__(self, C, stride,affine=True,dilation = 1,padding = 0 ):
    super(PointConv, self).__init__()
    if stride == 2:
      self.conv = nn.Conv1d(C,C*2, kernel_size=1, dilation = 1,stride=stride, bias=False)
    else:
      self.conv = nn.Conv1d(C,C, kernel_size=1, stride=stride,dilation = 1 , bias=False)

  def forward(self, x):
    x =self.conv(x)
    return x

class PointConvBR(nn.Module):
  def __init__(self, C, stride,affine=True,dilation = 1,padding = 0 ):
    super(PointConvBR, self).__init__()
    if stride == 2:
      self.conv = nn.Conv1d(C,C*2, kernel_size=1, dilation = 1,stride=stride, bias=False)
    else:
      self.conv = nn.Conv1d(C,C, kernel_size=1, stride=stride,dilation = 1 , bias=False)
    self.act =  nn.ReLU()
    self.bn = nn.BatchNorm1d(C, affine=affine)

  def forward(self, x):
    x = self.act(x)
    x =self.conv(x)
    x = self.bn(x)
    return x


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, length = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1)
        return x * y.expand_as(x)


class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MHA, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.pe = None



    def forward(self, x):
        batch_size = x.size(0)
        if self.pe == None or self.pe.shape[0] != 16:
          pe = torch.zeros(x.shape)
          pos = torch.arange(x.shape[2]).unsqueeze(1).view(1,-1)
          denominator = torch.exp(torch.arange(0,self.d_model,2)*(-math.log(10000.0)/self.d_model)).unsqueeze(1)
          pe[:,0::2,:] = torch.sin(pos *denominator)
          pe[:,1::2,:] = torch.cos(pos *denominator)
          self.pe = pe.cuda(x.device)
        x += self.pe
        # Calculate q, k, and v
        x = x.permute(0, 2, 1)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Split the last dimension into (heads, depth)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        output = self.out_linear(attn_output)
        output = output.permute(0, 2, 1)
        return output
