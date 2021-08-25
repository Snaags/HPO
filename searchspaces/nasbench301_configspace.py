import ConfigSpace as CS 
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import json as cs_json
import os


def nasbench301():
  current_dir = os.path.dirname(os.path.abspath(__file__))

  configspace_path = os.path.join(current_dir, 'configs/nasbench301_configspace.json')
  with open(configspace_path, "r") as f:
      json_string = f.read()
      configspace = cs_json.read(json_string)
      for i in configspace.get_conditions():
        print(i)
      input()
  return configspace 





if __name__ == "__main__":
  cs = nasbench301()
  print(cs.sample_configuration())   
