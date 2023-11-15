from .PlotItem import * 

import numpy as np

#
# ImageMetadataToScatterItem 
#
class ImageMetadataToScatterItem(PlotItem):

    def __init__(self):
        super().__init__(
          inputs={
            'images' : [], 
            'x'     : 'none',
            'y'     : 'none',
            'pencolor'  : 'default',
            'penwidth'  : 1.0,
            'brushcolor': 'default',
            'symbol'    : 'x',
            'size'      : 1.0 
          },
          outputs={
            'plotitem' : {} 
          }
        )

    def _update(self):

        xdata = []
        ydata = []
        xlabel = self.inputs.x.get()
        ylabel = self.inputs.y.get()
        for image in self.inputs.images.get():
            xdata.append(image.meta[xlabel])
            ydata.append(image.meta[ylabel])

        cleanData = self._castAndCleanData([xdata, ydata])

        out = { 'x' : {
                        'label' : self.inputs.x.get(),
                        'data'  : cleanData[0]
                      },
                'y' : {
                        'label' : self.inputs.y.get(),
                        'data'  : cleanData[1]
                      },
                'pen' : {
                            'color' : self.inputs.pencolor.get(), 
                            'width' : self.inputs.penwidth.get(),
                        },
                'brush' : {
                            'color' : self.inputs.brushcolor.get() 
                          },
                'symbol': self.inputs.symbol.get(),
                'size'  : self.inputs.size.get() 
              }
        self.outputs.plotitem.set({})
        self.outputs.plotitem.set(out)

        return 1
