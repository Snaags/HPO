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



OPS = {
  "gelu" : lambda C,kernel, stride,dil, affine: nn.GELU(),
  "relu" : lambda : nn.ReLU(),
  "batch_norm" : lambda C : nn.BatchNorm1d(C),
  "skip_connect" : lambda C : Identity(),
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
  
  
  }
class MHA(nn.Module):
  def __init__(self,C):
    super(MHA,self).__init__()
    self.act = nn.MultiheadAttention(C, C, batch_first = True)
  def forward(self,x):
    x = x.swapaxes(1,2)
    return self.act(x,x,x,need_weights=False)[0].swapaxes(1,2)
  

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

class DownsampleResolution(nn.Module):
  """
  This operation uses a depthwise convolution to reduce the signal resolution when needed.
  Given a stride of 2 its a kernel of 2 etc. 
  """
  def __init__(self, c_in,stride):
    super(DownsampleResolution,self).__init__()
    self.conv1 = nn.Conv1d(c_in,c_in, kernel_size =stride, stride = stride, bias = False, groups = c_in)
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
