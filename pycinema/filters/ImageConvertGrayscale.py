from pycinema import Filter

import PIL
import numpy

class ImageConvertGrayscale(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'images': []
          },
          outputs={
            'images': []
          }
        )

    def _update(self):
        results = []

        for image in self.inputs.images.get():
            rgbImage = PIL.Image.fromarray(image.channels['rgba'])
            grayscale = rgbImage.convert('LA').convert('RGBA') 

            outImage = image.copy()
            outImage.channels['rgba'] = numpy.array(grayscale)  
            results.append(outImage)

        self.outputs.images.set(results)

        return 1;
