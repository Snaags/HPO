import torch
import numpy

def freeze_normal_cells(model):

  for cell in model.normal_cells:
    for mList in cell.compute_order:
      for operation in mList.compute:
        for param in operation.parameters():
          param.requires_grad = False
  return model
          
def freeze_reduction_cells(model):
  for cell in model.reduction_cells:
    for mList in cell.compute_order:
      for operation in mList.compute:
        for param in operation.parameters():
          param.requires_grad = False

  return model

def freeze_all_cells( model ):
  model = freeze_reduction_cells( model )
  model = freeze_normal_cells( model )
  return model