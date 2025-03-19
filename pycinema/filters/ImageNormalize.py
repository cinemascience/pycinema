import numpy

from pycinema import Filter

class ImageNormalize(Filter):

    def __init__(self):
        super().__init__(
            inputs={
                'images': []
            },
            outputs= {
                'images': []
            }
        )

    def _update(self):
        results = []
        min = numpy.amin(numpy.stack(self.inputs.images.get()))
        max = numpy.amax(numpy.stack(self.inputs.images.get()))

        for image in self.inputs.images.get():
            image = (image - min) / (max - min)
            results.append(image)

        self.outputs.images.set(results)
        return 1
