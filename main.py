from HPO.algorithms.regevo_DARTS import main as  _algorithm
from HPO.workers.repsol_worker_cv import compute as _worker
from HPO.searchspaces.DARTS_rerun_config import init_config as _config

config = _config()
worker = _worker
algorithm = _algorithm

##Settings 

result = algorithm(worker, config)



