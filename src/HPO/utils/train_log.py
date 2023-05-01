import numpy as np 
import pandas as pd
import os

import os
import sqlite3
import time
class Logger:
    def __init__(self, path, experiment_name,dataset,fold,repeat,parameters):

        self.fold = fold
        self.parameters = parameters
        self.repeat = repeat
        self.experiment = experiment_name
        self.dataset = dataset
        # Set up the database connection
        self.conn = sqlite3.connect(path)
        self.c = self.conn.cursor()


    def update(self, data_dict):
        start_time = time.time()
        timeout=5
        retry_interval=0.05
        while time.time() - start_time < timeout:
            try:
                # Insert the provided data into the training_info table
                self.c.execute("""
                INSERT OR REPLACE INTO training_info (experiment, dataset,  model_id, epoch, training_loss, validation_loss, training_accuracy, validation_accuracy, learning_rate, confusion_matrix_train,confusion_matrix_test,fold,repeat,parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,? ,? ,? )
                """, (self.experiment, 
                  self.dataset, 
                  data_dict["ID"], 
                  data_dict["epoch"], 
                  data_dict["loss"], 
                  data_dict["validation_loss"], 
                  data_dict["training_accuracy"], 
                  data_dict["validation_accuracy"], 
                  data_dict["lr"],
                  data_dict["confusion_matrix_train"],
                  data_dict["confusion_matrix_test"],
                self.fold, 
                self.repeat,
                self.parameters
                )
                )
                self.conn.commit()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    # Wait for the retry interval before attempting again
                    time.sleep(retry_interval)
                    continue
                else:
                    # Raise the exception if it's not related to a locked database
                    raise e


    def close(self):
        # Close the database connection when done
        self.conn.close()