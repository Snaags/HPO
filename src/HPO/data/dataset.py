from HPO.data.HAR import *
from HPO.data.SHAR import *
from HPO.data.TEPS import *
from HPO.data.EEG import *
from HPO.data.FORDB import *
from HPO.data.UEA_datasets import *
DATASETS = {
    "teps" : (Train_TEPS,Test_TEPS),
    "HAR" : (Train_HAR,Test_HAR),
    "SHAR" : (SHAR, SHAR_TEST),
    "EEG" : (Train_EEG,Test_EEG),
    "LSST" : (Train_LSST,Test_LSST),
    "PhonemeSpectra" : (Train_PhonemeSpectra,Test_PhonemeSpectra),
    "FaceDetection" : (Train_FaceDetection,Test_FaceDetection),
    "PenDigits" : (Train_PenDigits,Test_PenDigits),
    "FORDB" : (Train_FORDB,Test_FORDB),
    "FaceDetection" : (Train_FaceDetection,Test_FaceDetection),
    "Full_FaceDetection" : (Full_FaceDetection,Full_FaceDetection),
    "Full_LSST" : (Full_LSST,Full_LSST),
    "Full_PhonemeSpectra" : (Full_PhonemeSpectra,Full_PhonemeSpectra),
    "UWaveGestureLibrary" : (Train_UWaveGestureLibrary,Test_UWaveGestureLibrary),
    "Full_UWaveGestureLibrary" : (Full_UWaveGestureLibrary,Full_UWaveGestureLibrary)
    
}


def get_dataset(name,train_args,test_args):
    if test_args == None:
        return DATASETS[name][0](**train_args)
    elif name in DATASETS:
        return DATASETS[name][0](**train_args),DATASETS[name][1](**test_args)
    else:
        return UEA_Full(name,**train_args) , None

