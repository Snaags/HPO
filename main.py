from HPO.algorithms.regevo_DARTS import main as  _algorithm
from HPO.workers.repsol_worker_cv import compute as _worker
from HPO.searchspaces.DARTS_NAS_config import init_config as _config
from HPO.algorithms.meta_cv import MetaCV
from HPO.workers.repsol_validate import _compute as validate
config = _config()
worker = _worker
algorithm = _algorithm
from multiprocessing import Pool
import numpy as np
import logging
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
##Hyperparameters for NAS 
hpo = {
  "batch_size" : 2,
  "channels" : 27,
  'jitter': 0.12412584247629389, 'jitter_rate': 0.5439942968995378, 'mix_up': 0.19412584247629389, 'mix_up_rate': 0.5439942968995378,
  'cut_mix': 0.19412584247629389, 'cut_mix_rate': 0.5439942968995378,'cut_out': 0.09412584247629389, 'cut_out_rate': 0.7439942968995378,
  'crop': 0.19412584247629389, 'crop_rate': 0.5439942968995378,
  'scaling': 0.001317169415702424, 'scaling_rate': 0.43534309734597858, 'window_warp_num': 3, 'window_warp_rate': 1.40015481616041954,
  'lr': 0.005170869707739693, 'p': 0.00296905723528657, 
  "epochs" : 70,
  "layers" : 3}
hpo = {
  "batch_size" : 2,
  "channels" : 27,
  'lr': 0.005170869707739693, 'p': 0.0, 
  "epochs" : 50,
  "layers" : 3}
##Settings 
budget = 8
logging.info("Starting MetaCV")
metacv = MetaCV(source_path = "/home/cmackinnon/scripts/datasets/repsol_mixed", destination_path = "/home/cmackinnon/scripts/datasets/repsol-meta-cv", num_samples = 87, num_folds = 5)
for i in range(5):
  metacv.next_fold()
  best_config, best_score , best_rec= algorithm(worker, config, "random_{}.csv".format(metacv.current_fold))
  best_config.update(hpo)
  with Pool(budget) as pool:
    res = pool.starmap(validate, [[best_config,1]]*8)
  acc = np.mean(np.asarray(res)[:,0])
  rec = np.mean(np.asarray(res)[:,1])
  print("Validated results for meta CV fold {}:".format(i))
  logging.info("Validated results for meta CV fold {}:".format(i))
  print("HPO CV score - ACC: {} - REC: {}".format(best_score,best_rec))
  logging.info("HPO CV score - ACC: {} - REC: {}".format(best_score,best_rec))
  print("MetaCV test set score - ACC: {} - REC: {}".format(acc,rec))
  logging.info("MetaCV test set score - ACC: {} - REC: {}".format(acc,rec))
