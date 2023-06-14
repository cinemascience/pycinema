from pycinema import Filter

import PIL
import numpy

class Border(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'width': 10,
            'color': 'AUTO'
          },
          outputs={
            'images': []
          }
        )

    def _update(self):

        images = self.inputs.images.get()

        results = []
        if len(images)<1:
            self.outputs.images.set(results)
            return 1

        color = self.inputs.color.get()
        if color=='AUTO':
            mean = images[0].channels['rgba'].mean(axis=(0,1))
            if (mean[0]+mean[1]+mean[2])/3<128:
                color = (255,255,255)
            else:
                color = (0,0,0)

        for image in images:
            # copy the input image
            rgba = image.channels['rgba']
            rgbImage = PIL.Image.fromarray( rgba )

            I1 = PIL.ImageDraw.Draw(rgbImage)
            shape = ((0,0), (rgba.shape[1] - 1, rgba.shape[0] - 1))
            I1.rectangle( shape, outline=color, width = self.inputs.width.get() )

            outImage = image.copy()
            outImage.channels['rgba'] = numpy.array(rgbImage)
            results.append( outImage )

        self.outputs.images.set(results)

        return 1
