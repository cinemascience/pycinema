from pycinema import Filter

import PIL
from PIL import ImageFilter
import numpy

class ImageFilterPIL(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'filter': 'BLUR', 
            'images': []
          },
          outputs={
            'images': []
          }
        )

    def _update(self):
        results = []

        # get the filter
        imFilter = ImageFilter.BLUR
        filterType = self.inputs.filter.get()
        if filterType == 'CONTOUR': 
            imFilter = ImageFilter.CONTOUR
        elif filterType == 'DETAIL': 
            imFilter = ImageFilter.DETAIL
        elif filterType == 'EDGE_ENHANCE': 
            imFilter = ImageFilter.EDGE_ENHANCE
        elif filterType == 'EDGE_ENHANCE_MORE': 
            imFilter = ImageFilter.EDGE_ENHANCE_MORE
        elif filterType == 'EMBOSS': 
            imFilter = ImageFilter.EMBOSS
        elif filterType == 'FIND_EDGES': 
            imFilter = ImageFilter.FIND_EDGES
        elif filterType == 'SHARPEN': 
            imFilter = ImageFilter.SHARPEN
        elif filterType == 'SMOOTH': 
            imFilter = ImageFilter.SMOOTH
        elif filterType == 'SMOOTH_MORE': 
            imFilter = ImageFilter.SMOOTH_MORE
        elif filterType == 'BLUR': 
            imFilter = ImageFilter.BLUR
        else:
            print("FILTER NOT FOUND")

        for image in self.inputs.images.get():
            rgbImage = PIL.Image.fromarray(image.channels['rgba'])
            filtered = rgbImage.filter(imFilter)

            outImage = image.copy()
            outImage.channels['rgba'] = numpy.array(filtered)  
            results.append(outImage)

        self.outputs.images.set(results)

        return 1;
