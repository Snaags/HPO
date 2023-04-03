from HPO.data.HAR import *
from HPO.data.SHAR import *
from HPO.data.TEPS import *
from HPO.data.EEG import *
from HPO.data.UEA_datasets import *
DATASETS = {
    "teps" : (Train_TEPS,Test_TEPS),
    "HAR" : (Train_HAR,Test_HAR),
    "SHAR" : (SHAR, None),
    "EEG" : (Train_EEG,Test_EEG)
}


def get_dataset(name,train_args,test_args):
    if test_args == None:
        return DATASETS[name][0](**train_args), None
    elif name in DATASETS:
        return DATASETS[name][0](**train_args),DATASETS[name][1](**test_args)
    else:
        return UEA_Full(name,**train_args) , None

