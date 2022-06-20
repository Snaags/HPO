from HPO.workers.TEPS_benchmark import _compute
import csv
from HPO.searchspaces.DARTS_config import init_config
hpo = {'batch_size': 2, 'channels': 64, 'jitter': 0.01241258424762939, 'jitter_rate': 0.5439942968995378, 'mix_up': 0.19412584247629389, 'mix_up_rate': 0.5439942968995378, 'cut_mix': 0.19412584247629389, 'cut_mix_rate': 0.5439942968995378, 'cut_out': 0.0941258424762939, 'cut_out_rate': 0.7439942968995378, 'crop': 0.19412584247629389, 'crop_rate': 0.5439942968995378, 'scaling': 0.001317169415702424, 'scaling_rate': 0.4353430973459786, 'window_warp_num': 3, 'window_warp_rate': 1.4001548161604196, 'lr': 0.0025170869707739693, 'p': 0.00, 'epochs': 10, 'layers': 3}
  #0.8125,0.7931034482758621,"
hyperparameter = {'normal_index_0_0': 0, 'normal_index_0_1': 0, 'normal_index_1_0': 2, 'normal_index_1_1': 0, 'normal_index_2_0': 0, 'normal_index_2_1': 0, 'normal_index_3_0': 2, 'normal_index_3_1': 2, 'normal_node_0_0': 'avg_pool_3x3', 'normal_node_0_1': 'none', 'normal_node_1_0': 'skip_connect', 'normal_node_1_1': 'max_pool_3x3', 'normal_node_2_0': 'sep_conv_5x5', 'normal_node_2_1': 'none', 'normal_node_3_0': 'avg_pool_3x3', 'normal_node_3_1': 'dil_conv_3x3', 'reduction_index_0_0': 0, 'reduction_index_0_1': 1, 'reduction_index_1_0': 2, 'reduction_index_1_1': 0, 'reduction_index_2_0': 3, 'reduction_index_2_1': 1, 'reduction_index_3_0': 1, 'reduction_index_3_1': 2, 'reduction_node_0_0': 'skip_connect', 'reduction_node_0_1': 'none', 'reduction_node_1_0': 'max_pool_3x3', 'reduction_node_1_1': 'avg_pool_3x3', 'reduction_node_2_0': 'skip_connect', 'reduction_node_2_1': 'sep_conv_5x5', 'reduction_node_3_0': 'sep_conv_5x5', 'reduction_node_3_1': 'sep_conv_3x3'}

  #0.8170731707317073,0.6486486486486487,"
hyperparameter = {'normal_index_0_0': 0, 'normal_index_0_1': 0, 'normal_index_1_0': 1, 'normal_index_1_1': 1, 'normal_index_2_0': 3, 'normal_index_2_1': 3, 'normal_index_3_0': 2, 'normal_index_3_1': 2, 'normal_node_0_0': 'avg_pool_3x3', 'normal_node_0_1': 'sep_conv_7x7', 'normal_node_1_0': 'sep_conv_5x5', 'normal_node_1_1': 'sep_conv_5x5', 'normal_node_2_0': 'sep_conv_7x7', 'normal_node_2_1': 'avg_pool_3x3', 'normal_node_3_0': 'skip_connect', 'normal_node_3_1': 'sep_conv_5x5', 'reduction_index_0_0': 1, 'reduction_index_0_1': 0, 'reduction_index_1_0': 0, 'reduction_index_1_1': 1, 'reduction_index_2_0': 1, 'reduction_index_2_1': 2, 'reduction_index_3_0': 4, 'reduction_index_3_1': 1, 'reduction_node_0_0': 'sep_conv_3x3', 'reduction_node_0_1': 'dil_conv_5x5', 'reduction_node_1_0': 'none', 'reduction_node_1_1': 'max_pool_3x3', 'reduction_node_2_0': 'dil_conv_3x3', 'reduction_node_2_1': 'avg_pool_3x3', 'reduction_node_3_0': 'none', 'reduction_node_3_1': 'sep_conv_5x5'}
cs = init_config()

def list2csv(list_, filename):
  with open(filename, "a+") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(list_) 

def main():
  while True:
    hp = cs.sample_configuration().get_dictionary()
    #hp.update(hpo)
    acc, rec ,sup ,unsup = _compute(hp, binary = False)
    list2csv([acc,rec,sup,unsup],"TEPS_eval.csv")

def dist():
  for i in range(1):
    hp = cs.sample_configuration().get_dictionary()
    #hp.update(hpo)
    acc, rec ,sup ,unsup,df_loss = _compute(hp, binary = False)
    list2csv([acc,rec,sup,unsup,i],"TEPS_aug.csv")
    df_loss.to_csv("loss_aug_{}.csv".format(i))
def nas():
  for i in range(100):
    hpo = {'batch_size': 32, 'channels': 32, 'jitter': 0.01241258424762939, 'jitter_rate': 0.5439942968995378, 'mix_up': 0.19412584247629389, 'mix_up_rate': 0.5439942968995378, 'cut_mix': 0.19412584247629389, 'cut_mix_rate': 0.5439942968995378, 'cut_out': 0.0941258424762939, 'cut_out_rate': 0.7439942968995378, 'crop': 0.19412584247629389, 'crop_rate': 0.5439942968995378, 'scaling': 0.001317169415702424, 'scaling_rate': 0.4353430973459786, 'window_warp_num': 3, 'window_warp_rate': 1.4001548161604196, 'lr': 0.00025170869707739693, 'p': 0.00, 'epochs': 30, 'layers': 6}
    hp = cs.sample_configuration().get_dictionary()
    hp.update(hpo)
    acc, rec ,sup ,unsup,cons, cons_10 = _compute(hp, binary = False)
    list2csv([acc,rec,sup,unsup,cons,cons_10,i],"TEPS_deep_limited.csv")

if __name__ == "__main__":
  nas() 
