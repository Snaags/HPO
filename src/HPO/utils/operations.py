import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
  'avg_pool_31x31' : lambda C, stride, affine: nn.AvgPool1d(31, stride=stride, padding=15, count_include_pad=False),
  'max_pool_31x31' : lambda C, stride, affine: nn.MaxPool1d(31, stride=stride, padding=15),
  'avg_pool_64x64' : lambda C, stride, affine: nn.AvgPool1d(63, stride=stride, padding=31, count_include_pad=False),
  'max_pool_64x64' : lambda C, stride, affine: nn.MaxPool1d(63, stride=stride, padding=31),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'memory_cell'  : lambda C, stride, affine: MemoryCell(),
  'point_conv' : lambda C, stride, affine: PointConv(C, stride,affine = affine),
  'depth_conv_7' :  lambda C, stride, affine: DepthConv(C,7, stride = stride ,padding = 3, affine = affine),
  'depth_conv_15' :  lambda C, stride, affine: DepthConv(C,15, stride = stride ,padding = 7, affine = affine),
  'depth_conv_29' :  lambda C, stride, affine: DepthConv(C,29, stride = stride ,padding = 14, affine = affine),
  'depth_conv_61' :  lambda C, stride, affine: DepthConv(C,61, stride = stride ,padding = 30, affine = affine),
  'depth_conv_101' :  lambda C, stride, affine: DepthConv(C,101, stride = stride ,padding = 50, affine = affine),
  'depth_conv_201' :  lambda C, stride, affine: DepthConv(C,201, stride = stride ,padding = 100, affine = affine),
  'patch_29' : lambda C, stride, affine: Patches(C,29,stride,padding = 14,affine = affine),
  'patch_61' : lambda C, stride, affine: Patches(C,61,stride,padding = 30,affine = affine),
  'SE_8' : lambda C, stride, affine: SE(C,8,stride = stride,affine = affine),
  'SE_16' : lambda C, stride, affine: SE(C,16,stride = stride,affine = affine),
  'SE_32' : lambda C, stride, affine: SE(C,32,stride = stride,affine = affine),
  'channel_mlp' : lambda C, stride, affine: MLPChannel(C,2,stride = stride,affine = affine),
  'attention_channel' : lambda C, stride, affine: AttentionChannel(C,stride = stride),
  'attention_space' : lambda C, stride, affine: AttentionSpace(C,stride = stride),
  
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'sep_conv_15x15' : lambda C, stride, affine: SepConv(C, C, 15, stride, 7, affine=affine),
  'sep_conv_30x30' : lambda C, stride, affine: SepConv(C, C, 27, stride, 13, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'dil_conv_7x7' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
}
class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm1d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class Patches(nn.Module):
  def __init__(self, c,patch_size ,stride, padding, affine=True):
    super(Patches, self).__init__()
    self.conv = nn.Conv1d(c,c*patch_size, kernel_size=patch_size, stride=patch_size*stride, padding=padding, bias=False)
    self.act = nn.GELU()
    self.norm = nn.BatchNorm1d(c, affine=affine)

  def forward(self, x):
    x =self.conv(x)
    x = self.act(x)
    x = self.norm(x)
    return x

class MixMLP(nn.Module):
  def __init__(self, C, stride,patch_size,affine = False):
    super(MixMLP,self).__init__()
 
    #self.norm = nn.LayerNorm()
    embedded_size = C*patch_size
    self.patch_size = patch_size
    self.mlp1 = nn.Linear(C*patch_size, embedded_size)
    self.ge = nn.GELU()
    self.mlp2 = nn.Linear(embedded_size, (C*patch_size)/stride)
  def forward(self, x):
    #Patches
    num_patches = x.shape[2]/self.patch_size
    patches = torch.chunk(x, num_patches,dim = 2)
    
    self.mlp1 

#self.patch = Patches(c,p,stride,padding)
class Attention(nn.Module):
  def __init__(self,max_len = 960,embedding = 200):
    super(Attention,self).__init__()
    self.sqrt_dk = np.sqrt(max_len)
    self.max_len = max_len
    self.ge = nn.GELU()
    self.q_w = nn.Linear(max_len, embedding,bias = False)
    self.v_w = nn.Linear(max_len, embedding,bias = False)
    self.k_w = nn.Linear(max_len, embedding,bias = False)
  def forward(self,x):
    pad_length = self.max_len - x.shape[2]
    x = F.pad(input = x, pad = (0,pad_length),value = 0)
    query = self.ge(self.q_w(x))
    key = self.ge(self.k_w(x))
    value = self.ge(self.v_w(x))
    score = torch.bmm(query,key.transpose(1,2))/self.sqrt_dk
    attn = nn.functional.softmax(score, dim = -1)
    out = torch.bmm(attn,value)
    return out
    
class AttentionSpace(nn.Module):
  def __init__(self,C,stride):
    super(AttentionSpace,self).__init__()
    d_k = C

    self.q_w = nn.Conv1d(C, d_k,bias = False,groups = C,kernel_size = 7,padding = 3,stride = stride)
    self.v_w = nn.Conv1d(C, d_k,bias = False,groups = C,kernel_size = 7,padding = 3,stride = stride)
    self.k_w = nn.Conv1d(C, d_k,bias = False,groups = C,kernel_size = 7,padding = 3,stride = stride)
    self.sqrt_dk = np.sqrt(d_k)
  def forward(self,x):
    #print("Incoming shape of x: {}".format(x.shape))
    query = self.q_w(x)   
    key = self.k_w(x)   
    value = self.v_w(x)
    #print("Query: {}".format(query.shape))
    score = torch.bmm(query,key.transpose(1,2))/self.sqrt_dk
    attn = nn.functional.softmax(score, dim = -1)
    out = torch.bmm(attn,value)
    return out

class AttentionChannel(nn.Module):
  def __init__(self,C,stride):
    super(AttentionChannel,self).__init__()
    d_k = C

    self.q_w = nn.Conv1d(C, d_k,bias = False,kernel_size = 1,padding = 0,stride = stride)
    self.v_w = nn.Conv1d(C, d_k,bias = False,kernel_size = 1,padding = 0,stride = stride)
    self.k_w = nn.Conv1d(C, d_k,bias = False,kernel_size = 1,padding = 0,stride = stride)
    self.sqrt_dk = np.sqrt(d_k)
  def forward(self,x):
    #print("Incoming shape of x: {}".format(x.shape))
    query = self.q_w(x)   
    key = self.k_w(x)   
    value = self.v_w(x)
    #print("Query: {}".format(query.shape))
    score = torch.bmm(query,key.transpose(1,2))/self.sqrt_dk
    attn = nn.functional.softmax(score, dim = -1)
    out = torch.bmm(attn,value)
    return out
    
class SE(nn.Module):
  def __init__(self, C,r,stride,affine = True ):
    super(SE,self).__init__()
    self.GP = nn.AdaptiveAvgPool1d(1)
    self.fc1 = nn.Linear(C, C//r, bias = False)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(C//r, C ,bias = False)
    self.sig = nn.Sigmoid()
    self.stride = stride
  def forward(self,x):
    #Squeeze
    y = self.GP(x).squeeze()# [Batch,C]
    #torch.mean(x,axis = 2)  
    
    y = self.fc1(y)
    y = self.act(y)
    y = self.fc2(y)
    y = self.sig(y).unsqueeze(dim = 2)
    x = x[:,:,::self.stride] 
    return x* y.expand_as(x)

class MLPChannel(nn.Module):
  def __init__(self, C,r,stride,affine = True ):
    super(MLPChannel,self).__init__()
    self.GP = nn.AdaptiveAvgPool1d(1)
    self.fc1 = nn.Linear(C, C*r, bias = False)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(C*r, C ,bias = False)
    self.sig = nn.Sigmoid()
    self.stride = stride
    if self.stride > 1:
      self.reduce = nn.AvgPool1d(kernel_size = self.stride,stride = self.stride,padding = 0)
      
  def forward(self,x):
    if self.stride > 1:
      x = self.reduce(x)
    #Squeeze
    y = self.GP(x).squeeze()# [Batch,C]
    #torch.mean(x,axis = 2)  
    
    y = self.fc1(y)
    y = self.act(y)
    y = self.fc2(y)
    y = self.sig(y).unsqueeze(dim = 2)
    return x* y.expand_as(x)

class SE_mask(nn.Module):
  def __init__(self, C,r,stride,affine = True ):
    super(SE,self).__init__()
    self.GP = nn.AdaptiveAvgPool1d(1)
    self.fc1 = nn.Linear(C, C//r, bias = False)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(C//r, C ,bias = False)
    self.sig = nn.Sigmoid()
    self.stride = stride
  def forward(self,x):
    #Squeeze
    y = self.GP(x).squeeze()# [Batch,C]
    #torch.mean(x,axis = 2)  
    
    y = self.fc1(y)
    y = self.act(y)
    y = self.fc2(y)
    y = self.sig(y).unsqueeze(dim = 2)
    x = x[:,:,::self.stride] 
    return x * y.expand_as(x)


class DepthConv(nn.Module):
  def __init__(self, c, kernel_size, padding,stride, affine=True):
    super(DepthConv, self).__init__()
    self.conv = nn.Conv1d(c,c, kernel_size=kernel_size, stride=stride, padding=padding, groups=c, bias=False)
    self.act = nn.GELU()
    self.norm = nn.BatchNorm1d(c, affine=affine)

  def forward(self, x):
    x =self.conv(x)
    x = self.act(x)
    x = self.norm(x)
    return x

class PointConv(nn.Module):
  def __init__(self, C, stride,affine=True):
    super(PointConv, self).__init__()
    self.conv = nn.Conv1d(C,C, kernel_size=1, stride=stride, bias=False)
    self.act = nn.GELU()
    self.norm = nn.BatchNorm1d(C, affine=affine)

  def forward(self, x):
    x =self.conv(x)
    x = self.act(x)
    x = self.norm(x)
    return x

class MemoryCell(nn.Module):
  def __init__(self):
    super(MemoryCell,self).__init__()
    self.cell = None

  def forward(self,x):
    if self.cell == None:
      self.cell = torch.ones(size = x.shape)
    out = self.cell.clone()
    self.cell = x
    return out

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

"""
class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm1d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:])], dim=1)
    out = self.bn(out)
    return out
"""
class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        self.conv1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv1d(C_in, math.ceil(C_out / 2), 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x[:,:,:])], dim=1)
        out = self.bn(out)
        return out
