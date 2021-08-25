import algorithms.regevo_alt as _algorithm
from workers.repsol_worker import compute as _worker
from searchspaces.OOP_config import init_config as _config

config = _config()
worker = _worker
algorithm = _algorithm.main

##Settings

result = algorithm(worker, config)



