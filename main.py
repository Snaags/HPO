from HPO.algorithms.Random import main as  _algorithm
from HPO.workers.repsol_worker_cv import compute as _worker
from HPO.searchspaces.DARTS_cv_config import init_config as _config
from HPO.algorithms.meta_cv import MetaCV

config = _config()
worker = _worker
algorithm = _algorithm

##Settings 
metacv = MetaCV(source_path = "/home/cmackinnon/scripts/datasets/repsol_mixed", destination_path = "/home/cmackinnon/scripts/datasets/repsol-meta-cv", num_samples = 87, num_folds = 5)
for i in range(5):
  metacv.next_fold()
  result = algorithm(worker, config)



