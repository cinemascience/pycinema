from pycinema import Filter

import cv2
import numpy

class ImageFilterOCV(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'conversion': 'GRAY', 
            'images': []
          },
          outputs={
            'images': []
          }
        )

    def _update(self):
        results = []

        if self.inputs.conversion.get() == 'GRAY':
            conversion = cv2.COLOR_BGRA2GRAY)

        for image in self.inputs.images.get():
            # convert from pycinema to cv2 format
            converted = cv2.cvtColor(image.channels['rgba'], cv2.COLOR_RGBA2BGRA)

            # apply new conversion
            grayscale = cv2.cvtColor(converted, conversion)

            # convert back to pycinema
            filtered  = cv2.cvtColor(grayscale, cv2.COLOR_BGRA2RGBA)

            # add to output
            outImage = image.copy()
            outImage.channels['rgba'] = numpy.array(filtered)  
            results.append(outImage)

        self.outputs.images.set(results)

        return 1;
