from pycinema import Filter
import logging as log

from os.path import exists
import os
import tensorflow as tf
import csv

class MLTFReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': ''
          },
          outputs={
            'model': []
          }
        )

    def _update(self):

        modelPath = self.inputs.path.get()
        #modelDirectoryPath = modelDirectoryPath + "/" #in case missing

        if not modelPath: #if the path is empty
            log.error('ML Model or data.csv path empty')
            self.outputs.models.set([])
            return 0

        if not exists(modelPath):
            log.error('[ERROR] ML Model or data.csv not found:', modelPath)
            self.outputs.models.set([])
            return 0

        # load_model supports two formats that a tensorflow 
        # model can be saved as: .keras and HDF5
        try:
            models = []
            # if the path directly points to a TF model
            if modelPath.endswith(".h5") or modelPath.endswith(".keras"):
                model = tf.keras.models.load_model(modelPath)
                models.append(model)

            else:
                # if the path points to a data.csv file 
                # the csv file must have the first column as numerical order of files
                # the second column is the path to each model
                # parse through models directory and load a list of models
                table = []
                with open(modelPath, 'r+') as csvfile:
                    rows = csv.reader(csvfile, delimiter=',')
                    for row in rows:  
                        table.append(row) 
                table = table[1:]
                numModels = len(table)

                models = [None] * numModels
                for row in table:
                    parent = os.path.dirname(modelPath) + "/"
                    filePath = os.path.join(parent, row[1])
                    model = tf.keras.models.load_model(filePath)      
                    models[int(row[0])] = model
            
            #check if training configuration exists, if not give error
        except:
            log.error('[ERROR] Unable to open ML Model Directory')
            self.outputs.models.set([])
            return 0

        self.outputs.model.set(models)

        return 1