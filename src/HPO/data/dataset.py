from HPO.data.HAR import *
from HPO.data.SHAR import *
from HPO.data.TEPS import *
from HPO.data.EEG import *
from HPO.data.FORDB import *
from HPO.data.UEA_datasets import *
#from Transformers.datasets.androzoo_dl import *
#from transformers.datasets import *
DATASETS = {
    "teps" : (Train_TEPS,Test_TEPS),
    "HAR" : (Train_HAR,Test_HAR),
    "SHAR" : (SHAR, SHAR_TEST),
    "EEG" : (Train_EEG,Test_EEG),
    "EEG_Retrain" : (Train_EEG,Test_EEG),
    "LSST" : (Train_LSST,Test_LSST),
    "PhonemeSpectra" : (Train_PhonemeSpectra,Test_PhonemeSpectra),
    "FaceDetection" : (Train_FaceDetection,Test_FaceDetection),
    "PenDigits" : (Train_PenDigits,Test_PenDigits),
    "PenDigitsRetrain" : (Train_PenDigits,Validation_PenDigits),
    "FORDB" : (Train_FORDB,Test_FORDB),
    "FaceDetection" : (Train_FaceDetection,Test_FaceDetection),
    "CharacterTrajectories" : (Train_CharacterTrajectories,Test_CharacterTrajectories),
    "SpokenArabicDigits" : (Train_SpokenArabicDigits,Test_SpokenArabicDigits),
    "FaceDetectionRetrain" : (Train_FaceDetection,Validation_FaceDetection),
    "PhonemeSpectraRetrain" : (Train_PhonemeSpectra,Validation_PhonemeSpectra),
    "EthanolConcentration" : (Train_EthanolConcentration,Test_EthanolConcentration),
    
    "LSSTRetrain" : (Train_LSST,Validation_LSST),
    #"Hex" : (Train_Hex,Test_Hex),
    "FaceDetectionVal" : (Train_FaceDetection,Validation_FaceDetection),
    "FaceDetectionTest" : (Train_FaceDetection,True_Test_FaceDetection),
    "PenDigitsVal" : (Train_PenDigits,Validation_PenDigits),
    "PenDigitsTest" : (Train_PenDigits,True_Test_PenDigits),
    "Full_FaceDetection" : (Full_FaceDetection,Full_FaceDetection),
    "Full_SpokenArabicDigits" : (Full_SpokenArabicDigits,Full_SpokenArabicDigits),

    "Full_LSST" : (Full_LSST,Full_LSST),
    "Full_EthanolConcentration" : (Full_EthanolConcentration,Full_EthanolConcentration),
    "Full_PenDigits" : (Full_PenDigits,Full_PenDigits),
    "Full_PhonemeSpectra" : (Full_PhonemeSpectra,Full_PhonemeSpectra),
    "UWaveGestureLibrary" : (Train_UWaveGestureLibrary,Test_UWaveGestureLibrary),
    "Full_UWaveGestureLibrary" : (Full_UWaveGestureLibrary,Full_UWaveGestureLibrary)
    
}


def get_dataset(name,train_args,test_args):
    if name in DATASETS:
        if test_args == None:
            return DATASETS[name][0](**train_args)
        return DATASETS[name][0](**train_args),DATASETS[name][1](**test_args)
    else:
        if "Full" in name:
            return Full_N(name.split("_")[-1],**train_args)
        elif "Retrain" in name:
            return Train_N(name.split("_")[0],**train_args), Validation_N(name.split("_")[0],**test_args)
        else:
            return Train_N(name,**train_args), Test_N(name,**test_args)

