from HPO.algorithms.Random import main as  _algorithm
from HPO.workers.repsol_worker_full import compute as _worker
from HPO.searchspaces.cnn_config import init_config as _config

config = _config()
worker = _worker
algorithm = _algorithm

##Settings 

result = algorithm(worker, config)



