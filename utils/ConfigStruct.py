import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import copy

"""
Christopher MacKinnon 2021

The objective of this class is a layer of abstraction on top of the ConfigSpace package, allowing
parameters to be defined in a hierarchy rather than absolutely defining each parameter. This is 
aimed at allowing the hyperparameter for each layer in a model to be dynamically constructed once 
a layer is defined 
    
"""

class Parameter:
    """
    Base parameter class. This is effectively a wrapper around the ConfigSpace hyperparameters
    """ 
    def __init__(self, name : str , hyperparameter_type : str, lower_or_constant_value , upper_value = None ,normal = False, log : bool = False):
        self.name = name
        self.type = hyperparameter_type
        self.lower = lower_or_constant_value 
        self.upper = upper_value
        if self.lower == self.upper:
          self.type = "Constant"
          print("Warning: Set type as integer but range same upper and lower values given, Setting to Constant")
        self.normal = normal 
        self.log = log 
        if self.type == "Constant":
            self.config = CSH.Constant(name = name, value = lower_or_constant_value)
        if normal == False:
            if self.type == "Integer":
                self.config = CSH.UniformIntegerHyperparameter(name = name, lower = lower_or_constant_value, upper = upper_value, log = log)

            if self.type == "Float":
                self.config = CSH.UniformFloatHyperparameter(name = name, lower = lower_or_constant_value, upper = upper_value, log = log)
         
        if normal == True:
            if self.type == "Integer":
                self.config = CSH.NormalIntegerHyperparameter(name = name, lower = lower_or_constant_value, upper = upper_value, log = log)

            if self.type == "Float":
                self.config = CSH.NormalFloatHyperparameter(name = name, lower = lower_or_constant_value, upper = upper_value, log = log)
        if self.type == "Categorical":
          self.config = CSH.CategoricalHyperparameter(name = name , choices = lower_or_constant_value)

    def has_children(self):
        return False

    def has_unexpanded_children(self):
        return self.has_children()

    def get_args(self, args = None):
    
        return self.type, self.lower, self.upper, self.normal, self.log
    
    def set_arg_upper(self, value):
        self.upper = value

class LTP_Parameter(Parameter):
    #Less Than Parent, Sets upper limit to value of parent from string (i.e. conv_5_size.upper = 4)
  def __init__(self, name : str , hyperparameter_type : str, lower_or_constant_value , upper_value = None ,normal = False, log : bool = False):
    super().__init__( name , hyperparameter_type , lower_or_constant_value , upper_value ,normal , log )

  def get_args(self, parent_string = None):
    if parent_string != None:
      self.upper = int(parent_string[:-(len("_"+self.name))][-1]) - 1 
    return self.type, self.lower, self.upper, self.normal, self.log


class Integer_Struct(Parameter):
    """
    Generates 1 of each child parameter for each integer between lower and upper and adds conditions 
    name of children follows format parent_name + _ + number + _ + child_name
    """
    
    def __init__(self, config_space,  children : list,struct_name : str , name : str , hyperparameter_type : str, lower_or_constant_value , upper_value = None ,normal = False, log :  bool = False ):
        
        super().__init__(name, hyperparameter_type, lower_or_constant_value, upper_value, normal, log)
        self.children_template = children
        self.children = []
        self.children_dict = dict()
        self.conditions = []
        self.config_space = config_space
        self.struct_name = struct_name 
        self.expanded = False
    def add_children_to_config_space(self):
        self.config_space.add_hyperparameters( [child.config for child in self.children] )
        self.config_space.add_hyperparameter(self.config)

    def generate_name(self,parent_name : str, child_name : str, num : int):
        return parent_name + "_" + str(num) +"_" + child_name  

    def has_unexpanded_children(self):
      if self.expanded == False:
        self.expanded = True 
        return True    
      else:
        return False

    def expand_children(self):
        for i in self.children_template:
          if i.has_unexpanded_children():
            p_children = i.generate_children()
            self.children_template.extend(p_children)
            


    def _generate_child_set(self, i ):
        #Generates children for iteration i of parent
        children = list()
        
        for child_template in self.children_template:
            new_child_name = self.generate_name(self.struct_name, child_template.name , i )
            print(new_child_name , child_template.get_args())
            children.append( Parameter( new_child_name, *child_template.get_args(new_child_name) ))
        return children

    def _generate_conditions(self):
        pass

    def generate_children(self, rec = False):
        #Generate child parameter for each value of the parent class
        for i in range(self.lower, self.upper + 1): 
            children = self._generate_child_set(i)
            self.children += children
            self.children_dict[i] = children
        return self.children

class Cumulative_Integer_Struct(Integer_Struct):
    """ (CIS)
    Generates an instance for each child parameter for 1 to n (i.e. 1,2,..,n ) where the parent parameter 
    value is n.
    """
    def __init__(self,config_space ,  children : list,struct_name, name : str ,
                hyperparameter_type : str, lower_or_constant_value , 
                upper_value = None ,normal = False, log :bool = False ):

        super().__init__(config_space ,children, struct_name ,name, 
                hyperparameter_type, lower_or_constant_value, upper_value, normal, log)



     
    def batch_add_greater_than_cond(self, a_list, b , num):
        for a in a_list:
            if num != self.lower: 
                cond = CS.GreaterThanCondition(a.config,b,num -1 )
                self.config_space.add_condition(cond)  
    
    def generate_conditions(self):
        for i in range(self.lower, self.upper + 1):
            self.batch_add_greater_than_cond( self.children_dict[i] , self.config , i )
    def init(self):
        self.expand_children()
        self.generate_children()
        self.add_children_to_config_space()
        self.generate_conditions()



if __name__ == "__main__":
    pass        
