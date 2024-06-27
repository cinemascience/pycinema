from pycinema import Filter

from PIL import Image
import tensorflow
import numpy as np

# Machine Learning Prediction filter
# Reads a ML trained model and predicts the results based on 
# an input or set of inputs
class MLTFPredictor(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'trainedModels': [],
            'images': []
          },
          outputs={
            'images': [],
            # 'predictedValues': []
          }
        );

    def _update(self):
        images = self.inputs.images.get()
        modelList = self.inputs.trainedModels.get()
        # get required input properties from first model
        model = modelList[0]
        width = model.layers[0].input_shape[1]
        height = model.layers[0].input_shape[2]
        channels = model.layers[0].input_shape[3]
        if channels == 1:
          gray_req = True
        else: #channels == 3 or 4
          gray_req = False

        result = []
        # iterate over all the images in the input images
        for image in self.inputs.images.get():
          img = image.copy()
          data = image.channels['rgba']
          if gray_req:
            data = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])
          else:
            data = [[item[:3] for item in inner_list] for inner_list in data]             
          data = np.array(data) / 255
          data = data.reshape((1, width, height, channels))

          pv_iter = 0
          for model in modelList:
            # at the moment, assuming only one predicted output
            # from the network
            one_hot = model.predict(data, verbose = 0)
            predictedValue = np.argmax(one_hot)
            img.meta['PredictedValue_' + str(pv_iter)] = float(predictedValue)
            pv_iter += 1
          
          result.append(img)

        self.outputs.images.set(result)

        return 1;