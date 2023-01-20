
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

OPS = {
  'avg_pool_3_1' : lambda C,kernel, stride,dil, affine: nn.AvgPool1d((2**kernel)-1, stride=stride, padding = (2**kernel//2) -1, count_include_pad=False),
 
  'max_pool_3_1' : lambda C,kernel, stride,dil, affine: nn.MaxPool1d((2**kernel)-1, stride=stride,padding = ((2**kernel)//2) -1),
  'gelu' : lambda C,kernel, stride,dil, affine: GELU(),
  'batch_norm' : lambda C,kernel, stride,dil, affine: BatchNorm(C,affine = affine),
  'point_conv' : lambda C,kernel, stride,dil, affine: PointConv(C, stride ,affine = affine),
  'depth_conv' :  lambda C,kernel, stride, dil,affine: DepthConv(C,(2**kernel)-1, stride = stride ,dilation = 2**dil,padding = 3, affine = affine),
  'skip_connect' : lambda C, kernel, stride,dil, affine: Identity(),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),

  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
  #'avg_pool_31x31' : lambda C, stride, affine: nn.AvgPool1d(31, stride=stride, padding=15, count_include_pad=False),
  #'max_pool_31x31' : lambda C, stride, affine: nn.MaxPool1d(31, stride=stride, padding=15),
  #'avg_pool_64x64' : lambda C, stride, affine: nn.AvgPool1d(63, stride=stride, padding=31, count_include_pad=False),
  #'max_pool_64x64' : lambda C, stride, affine: nn.MaxPool1d(63, stride=stride, padding=31),
  #'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else PointConv(C,stride , affine=affine),
  #'point_conv' : lambda C, stride, affine: PointConv(C, stride,affine = affine),
  'depth_conv_7' :  lambda C, stride, affine: DepthConv(C,7, stride = stride ,padding = 3, affine = affine),
  #'depth_conv_15' :  lambda C, stride, affine: DepthConv(C,15, stride = stride ,padding = 7, affine = affine),
  #'depth_conv_29' :  lambda C, stride, affine: DepthConv(C,29, stride = stride ,padding = 14, affine = affine),
  #'depth_conv_61' :  lambda C, stride, affine: DepthConv(C,61, stride = stride ,padding = 30, affine = affine),
  #'depth_conv_101' :  lambda C, stride, affine: DepthConv(C,101, stride = stride ,padding = 50, affine = affine),
  #'depth_conv_201' :  lambda C, stride, affine: DepthConv(C,201, stride = stride ,padding = 100, affine = affine),
  #'gelu' : lambda C, stride, affine: GELU(),
  #'batch_norm' : lambda C, stride, affine: BatchNorm(C,affine = affine)
  }

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride].mul(0.)

class DownSampleResolution(nn.Module):
  """
  This operation uses a depthwise convolution to reduce the signal resolution when needed.
  Given a stride of 2 its a kernel of 2 etc. 
  """
  def __init__(self, stride, c_in):
    self.conv1 = nn.Conv1d(c_in,c_in*stride, kernel_size =kernel_size, stride = stride, bias = False, groups = c)
  def forward(self,x):
    return self.conv1(x)

class ReSampleChannels(nn.Module):
  """
  This operation uses a pointwise convolution to change the number of channels when needed.
  """
  def __init__(self, stride, c_in,c_out):
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
    y = self.fc2(y)
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
    padding = (kernel_size*stride*dilation)//2
    self.conv = nn.Conv1d(c,c, kernel_size=kernel_size, stride=stride, padding=padding, groups=c, bias=False)

  def forward(self, x):
    x =self.conv(x)
    return x

class PointConv(nn.Module):
  def __init__(self, C, stride,affine=True):
    super(PointConv, self).__init__()
    if stride == 2:
      self.conv = nn.Conv1d(C,C*2, kernel_size=1, stride=stride, bias=False)
    else:
      self.conv = nn.Conv1d(C,C, kernel_size=1, stride=stride, bias=False)

  def forward(self, x):
    x =self.conv(x)
    return x
