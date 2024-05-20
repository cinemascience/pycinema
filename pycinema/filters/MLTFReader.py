from pycinema import Filter
import logging as log

from os.path import exists
import tensorflow as tf

class MLTFReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': ''
          },
          outputs={
            'model': None
          }
        )

    def _update(self):

        modelPath = self.inputs.path.get()

        if not modelPath: #if the path is empty
            log.error('ML Model path empty')
            self.outputs.model.set(None)
            return 0

        if not exists(modelPath):
            log.error('[ERROR] ML Model not found:', modelPath)
            self.outputs.model.set(None)
            return 0

        try:
            # load_model supports the three formats that a tensorflow 
            # model can be saved as: .keras, HDF5 and SavedModel 
            model = tf.keras.models.load_model(modelPath)
            
            #check if training configuration exists, if not give error
        except:
            log.error('[ERROR] Unable to open ML Model')
            self.outputs.model.set(None)
            return 0

        self.outputs.model.set(model)

        return 1
