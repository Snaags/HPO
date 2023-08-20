from HPO.workers.ensemble import EnsembleManager
from multiprocessing import Process

def full_eval(SETTINGS,device):
  accuracy = {}
  #acc_best_single, recall,params = evaluate("{}/{}".format(SETTINGS["PATH"],"configuration.json"))
  for i in [1,3,5,10]:
    be = EnsembleManager("{}/{}".format(SETTINGS["PATH"],"configuration.json"),device)
    be.get_ensemble(i)
    accuracy["ensemble_{}".format(i)] = be.evaluate(2)
  be.get_ensemble(10)
  accuracy["distill"] = be.distill_model()
    
  # convert dictionary to dataframe
  df = pd.DataFrame(accuracy, index=[0])

  # save to csv
  df.to_csv('{}/test_results-2.csv'.format(SETTINGS["PATH"]), index=False)

if __name__ == "__main__":
    experiments
    glob = "results-drop"
    for i in os.listdir(path):
        if glob in i:
            experiments.append(i)

    gpus = [0,1,2,3]
    for i in experiments:
        full_eval(SETTINGS,device)

