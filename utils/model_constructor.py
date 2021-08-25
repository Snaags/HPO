import utils.ops1d as ops
import random
#hyperparameter processing
from operator import itemgetter 
import torch.nn as nn
import time
class DataShapeLogger:
  def __init__(self, filename):
    self.filename = filename
    self.log_list = [] 
    MyFile=open(self.filename+".txt",'a') 
    MyFile.write(str(time.time()))
    MyFile.write('\n')
  def log(self, *string):
    string = [str(x) for x in string ]
    out_string = "".join(string)
    MyFile=open(self.filename+".txt",'a')
    
    MyFile.write(out_string)
    MyFile.write('\n')
    MyFile.close()

class Model(nn.Module):
  def __init__(self, input_size, output_size, hyperparameters):
    super(Model,self).__init__()
    self.log_flag = True
    self.logger = DataShapeLogger("logger.txt")
    self.hyperparameters = hyperparameters  
    self.channels = hyperparameters["channels"]
    self.normal_cells = nn.ModuleList()
    self.reduction_cells = nn.ModuleList()
    self.p = hyperparameters["p"]
    self.layers = hyperparameters["layers"]
    self.in_conv = ops.StdConv(input_size[0], self.channels)
    self.build_cells(hyperparameters)
    self.gap = ops.AdaAvgPool() 
    self.fc_list = nn.ModuleList()
    channels = self.channels
    while channels > 2*output_size:
      self.fc_list.append(nn.Linear(channels,channels//2))
      self.fc_list.append(nn.ReLU())
      channels = channels // 2
    self.fc = nn.Linear(channels, output_size)
    self.outact = nn.Softmax(dim = 1)
  def _build_dict(self,parameters : dict, keyword : str):
    _dictionary = dict()
    keyword_length = len(keyword)
    id_index = keyword_length + 1
    
    for parameter in parameters:
      if parameter[:keyword_length] == keyword:
        cell_id = int(parameter[id_index])

        operation_key = parameter[id_index + 2 : ]
        operation_value = parameters[ parameter ]
        
        if cell_id in _dictionary.keys():        
          _dictionary[ cell_id ][ operation_key ] = operation_value
         
        else: #if dictionary doesnt exist, make it
          _dictionary[ cell_id ] = { operation_key : operation_value }

    return _dictionary
  
  def build_cells(self, parameters): 
    conv_dictionary = self._build_dict(parameters, "normal_cell")
    redu_dictionary = self._build_dict(parameters, "reduction_cell")
    
    for i in range(parameters["layers"]):
         
      self.normal_cells.append(Cell(conv_dictionary[1],self.channels,self.channels,p = self.p))
      if i < (self.layers -1):
        self.reduction_cells.append(Cell(redu_dictionary[1],self.channels,channels_out = self.channels*1, p = self.p))
        #self.channels*=2 
  
  def _forward(self,x):
    x = self.in_conv(x)
    for i in range(self.layers):
      x = self.normal_cells[i](x) 
      if i != (self.layers -1):
        x = self.reduction_cells[i](x)
    x = self.gap(x)
    x = x.squeeze()
    for i in self.fc_list:
      x = i(x)
    x = self.fc(x)
    x = self.outact(x)
    return x  

  def _forward_log(self,x):
    self.logger.log("Input Size: ", x.size())
    x = self.in_conv(x)
    self.logger.log("After in_conv: ", x.size() )
    for i in range(self.layers):
      x = self.normal_cells[i](x) 
      self.logger.log("Data after Normal Cell ", str(i),": ",x.size())
      if i != (self.layers -1):
        x = self.reduction_cells[i](x)
        self.logger.log("Data after Reduction Cell ", str(i),": ",x.size())
    self.logger.log("Size of x after cells: ", x.size())
    x = self.gap(x)
    self.logger.log("Size of x after gap: ", x.size())
    self.logger.log("Size of dense input: ", self.channels)
    x = x.squeeze()
    for i in self.fc_list:
      x = i(x)
    x = self.fc(x)
    x = self.outact(x)
    return x  

  def forward(self,x):
    if self.log_flag == True:
      self.log_flag = False
      return self._forward_log(x)
    else:
      return self._forward(x)

class Ops(nn.Module):
  def __init__(self, parameters, channels_in,channels_out, p):
    super(Ops,self).__init__()
    self.args = {}
    self.channels_in = channels_in
    self.channels_out = channels_out
    self.multicompute = False
    self.p = p
    self.input = []
    #self.dropout = nn.Dropout(p = 0.2)
    for i in parameters:
      if i == "type":
        self.op = parameters[i]
        self.operation = self.get_operation(parameters[i])
      elif i[:-2] == "input":
          if parameters[i] not in self.input:
            self.input.append(parameters[i])
      else:
          self.args[i] = parameters[i]
    self.compute = nn.ModuleList()
    if len(self.input) > 1:
      self.compute.append(ops.StdAdd())
      self.multicompute = True
      self.pool = nn.AvgPool1d(2)
    self.compute.append(self.operation(**self.args))
  def get_required(self) -> list:
    return self.input

  def get_operation(self, op_key):
    if op_key == "StdConv":
      operation = ops.StdConv
      self.args["C_in"] = self.channels_in 
      self.args["C_out"] = self.channels_out
      self.args["padding"] = "same" 
    elif op_key == "SepConv3":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 3
      self.args["stride"] = 1
      self.args["padding"] = "same"
      self.args["separable"] = True
      self.args["C_in"] = self.channels_in 
      self.args["C_out"] = self.channels_out
    elif op_key == "SepConv5":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 5
      self.args["stride"] = 1
      self.args["padding"] = "same" 
      self.args["separable"] = True
      self.args["C_in"] = self.channels_in 
      self.args["C_out"] = self.channels_out
    elif op_key == "SepConv7":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 7
      self.args["stride"] = 1
      self.args["padding"] = "same" 
      self.args["separable"] = True
      self.args["C_in"] = self.channels_in 
      self.args["C_out"] = self.channels_out
    elif op_key == "Conv3":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 3
      self.args["stride"] = 1
      self.args["padding"] = "same"
      self.args["separable"] = False
      self.args["C_in"] = self.channels_in 
      self.args["C_out"] = self.channels_out
    elif op_key == "Conv5":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 5
      self.args["stride"] = 1
      self.args["padding"] = "same" 
      self.args["separable"] = False
      self.args["C_in"] = self.channels_in 
      self.args["C_out"] = self.channels_out
    elif op_key == "Conv7":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 7
      self.args["stride"] = 1
      self.args["padding"] = "same" 
      self.args["separable"] = False
      self.args["C_in"] = self.channels_in 
      self.args["C_out"] = self.channels_out
    elif op_key == "MaxPool5":    
      operation = ops.Pool
      self.args["pool_type"] = "max"
      self.args["kernel_size"] = 5
      self.args["padding"] = 2
    elif op_key == "MaxPool7":    
      operation = ops.Pool
      self.args["pool_type"] = "max"
      self.args["kernel_size"] = 7
      self.args["padding"] = 3
    elif op_key == "AvgPool5":    
      operation = ops.Pool
      self.args["pool_type"] = "avg"
      self.args["kernel_size"] = 5
      self.args["padding"] = 2
    elif op_key == "AvgPool7":    
      operation = ops.Pool
      self.args["pool_type"] = "avg"
      self.args["kernel_size"] = 7
      self.args["padding"] = 3
    elif op_key == "Identity":    
      operation = ops.Identity
    elif op_key == "FactorizedReduce":
      operation = ops.FactorizedReduce
      self.args["C_in"] = self.channels_in 
      self.args["C_out"] = self.channels_out
    elif op_key == "":
      operation = ops.StdConv
    return operation
  def forward( self, x ):
    #print(self.op)
    for count,i in enumerate(self.compute):
      if self.multicompute and count == 0:
        #print("Size Before operation: ", x[0].size(),x[1].size())
        x = i(*x) 
        #print("Size After operation: ", x.size())
      else:
        #x = self.dropout(x)
        #print("Size Before operation: ", x.size())
        x = i(x)
        #print("Size After operation: ", x.size())
    #if self.multicompute:
      #x = self.pool(x)
    return x 
  
  def process( self, x : dict):
    return self.forward(itemgetter( *self.get_required() )( x ))  

class Cell(nn.Module):
  """
  Contains a series of operations and information links
  """
  def __init__(self,parameters,channels_in,channels_out,p):
    super(Cell, self).__init__()
    self.ops_id = [] #numerical identifier for each operation 
    self.ops = []
    self.p = p
    self.channels_in = channels_in
    self.channels_out = channels_out
    self.inputs = []

    self.build_ops(parameters)
    self.output_operation = max(self.ops_id) #parameters["num_ops"]
    self.compute_order = nn.ModuleList()
    self.cal_compute_order()
  def _build_dict(self,parameters : dict, keyword : str):
    _dictionary = dict()
    keyword_length = len(keyword)
    id_index = keyword_length + 2
    
    for parameter in parameters:
      if parameter[:keyword_length] == keyword:
        cell_id = int(parameter[id_index])

        operation_key = parameter[id_index + 2 : ]
        operation_value = parameters[ parameter ]
        
        if cell_id in _dictionary.keys():        
          _dictionary[ cell_id ][ operation_key ] = operation_value
         
        else: #if dictionary doesnt exist, make it
          _dictionary[ cell_id ] = { operation_key : operation_value }

    return _dictionary
  def build_ops(self, parameters):
    ops_dictionary = self._build_dict(parameters, "op")  
    for i in ops_dictionary:
      
      self.ops.append(Ops(ops_dictionary[i], self.channels_in,self.channels_out, self.p))
      self.ops_id.append(i)
  def cal_compute_order(self):
    #Calculate the order in which the operations in a cell should be computed
    self.com_order_id_list = [] #Tracks the ID of the operations

    current_operation = self.output_operation 
    self._compute_order(current_operation)
    
    ml = nn.ModuleList()
    self.new_compute_order = []
    for i in range(len(self.compute_order)-1,-1, -1):

      if self.compute_order[i] not in ml:
       
        self.new_compute_order.append(self.com_order_id_list[i]) # The final operation order ID list
        ml.append(self.compute_order[i])
    self.compute_order = ml



  def _compute_order(self,operation): 
      #Appends an operation to the compute list and then traverses the prerequisite 
      #operations recursively 
      self.com_order_id_list.append(operation)
      self.compute_order.append(self.ops[self.ops_id.index(operation)])


      #Get the requirements compute inputs as long as the input is not zero
      for i in self.ops[self.ops_id.index(operation)].get_required():
        if i != 0:
          self._compute_order(i)
      return 
    
    
  def forward(self, x):
    outputs = {0 : x}
    for id,op in zip(self.new_compute_order,self.compute_order):
      outputs[id] = op.process(outputs)

    return outputs[self.output_operation] 

 
