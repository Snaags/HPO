import os
import random
import time
import shutil
from sklearn.model_selection import StratifiedKFold as KFold
class MetaCV:
  def __init__(self, source_path, destination_path, num_samples, num_folds = 3, log_file = "log.txt"):
    assert(num_samples == len(os.listdir(source_path))), "Number given samples {} does not match number of files in directory {}".format(num_samples,len(os.listdir(source_path)))
    fold_size = int(num_samples/num_folds)
    self.source_path = source_path
    self.destination_path = destination_path
    self.log_file = log_file
    self.current_fold = 0
    self.sample_files = os.listdir(source_path)
    self.num_folds = num_folds
    self.seed = random.randint(0,9999)
    #This is where the active samples will be stored.
    self.main_train_path = "{}/train".format(self.destination_path)
    self.main_test_path = "{}/test".format(self.destination_path)
    if not os.path.exists(destination_path):
      os.mkdir(destination_path)
    if not os.path.exists(self.main_train_path):
      os.mkdir(self.main_train_path) 
    if not os.path.exists(self.main_test_path):
      os.mkdir(self.main_test_path) 
    self.get_classes() 
    self.kfold = KFold(n_splits = self.num_folds, shuffle = True,random_state = self.seed)
    self.fold = self.kfold.split(self.sample_files,self.class_list)

  def get_classes(self):
    self.class_list = []
    for i in self.sample_files:
      self.class_list.append(i[3:11])
      
    print("List of classes: {}".format(set(self.class_list)))
 
  def next_fold(self):
    #Set up storing of samples
    assert(self.current_fold < self.num_folds), "[Error]: Tried to access fold {} when there are only {} folds!".format(self.current_fold, self.num_folds)
    self.check_for_overwrite(self.destination_path,self.current_fold)
    print("Starting Meta Cross Validation fold: {}".format(self.current_fold))
    train_path = "{}/cv-{}/train".format(self.destination_path,self.current_fold)
    test_path = "{}/cv-{}/test".format(self.destination_path,self.current_fold)
    self.clear_main_dir()
    os.mkdir("{}/cv-{}".format(self.destination_path,self.current_fold)) 
    os.mkdir(train_path) 
    os.mkdir(test_path) 
    current_fold_train_indexes, current_fold_test_indexes = next(self.fold)
    
    for file_index in current_fold_train_indexes:
      self.move_file(train_path,file_index)
      self.move_file(self.main_train_path,file_index)

    for file_index in current_fold_test_indexes:
      self.move_file(test_path,file_index)
      self.move_file(self.main_test_path,file_index)
    self.current_fold += 1

  def clear_main_dir(self):
      train_files = os.listdir(self.main_train_path)
      test_files = os.listdir(self.main_test_path)
      for i in train_files:
        os.remove("{}/{}".format(self.main_train_path,i)) 
      for i in test_files:
        os.remove("{}/{}".format(self.main_test_path,i)) 
       
  def move_file(self,path, idx):
      shutil.copyfile("{}/{}".format(self.source_path,self.sample_files[idx]) , "{}/{}".format(path,self.sample_files[idx]))

  def check_for_overwrite(self,dest, fold):
    while fold < self.num_folds:
      if os.path.exists("{}/cv-{}".format(dest, fold)):
        ans = input("folder for cv-{} already exists overwrite? [Y/n]".format(fold))
        if ans.upper() == "Y":
          if os.path.exists("{}/cv-{}-old".format(dest, fold)):
            os.system("rm -r {}/cv-{}-old".format(dest, fold))
          os.rename("{}/cv-{}".format(dest, fold),"{}/cv-{}-old".format(dest, fold))
          with open("{}/{}".format(self.destination_path, self.log_file), "w") as text_file:
            text_file.write(str(self.seed))
        else:
          print("Loading previous run")
          with open("{}/{}".format(self.destination_path, self.log_file), "r") as text_file:
            self.seed = int(text_file.read())
            self.kfold = KFold(n_splits = self.num_folds, shuffle = True,random_state = self.seed)
            self.fold = self.kfold.split(self.sample_files,self.class_list)
            folds = os.listdir(self.destination_path)
            max_val = 0
            for i in folds:
              if i[-1].isnumeric:
                if int(i[-1]) > max_val:
                  max_val = int(i[-1])
            self.current_fold = max_val

            
      else:
        with open("{}/{}".format(self.destination_path, self.log_file), "w") as text_file:
          text_file.write(str(self.seed))
      fold += 1
