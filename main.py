import HPO.algorithms.regevo_alt as _algorithm
from HPO.workers.repsol_worker_full import compute as _worker

from HPO.searchspaces.OOP_config import init_config as _config

config = _config()
worker = _worker
algorithm = _algorithm.main

##Settings 

result = algorithm(worker, config)



