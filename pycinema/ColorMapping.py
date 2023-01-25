from .Core import *
import numpy
import matplotlib.cm as cm

class ColorMapping(Filter):

    def __init__(self):
        super().__init__()

        self.addInputPort("map", "plasma")
        self.addInputPort("nan", (0,0,0,0))
        self.addInputPort("range", (0,1))
        self.addInputPort("channel", "depth")
        self.addInputPort("images", [])
        self.addOutputPort("images", [])

    def update(self):

        images = self.inputs.images.get()

        iChannel = self.inputs.channel.get()

        results = []

        cmap = cm.get_cmap( self.inputs.map.get() )
        cmap.set_bad(color=self.inputs.nan.get() )
        r = self.inputs.range.get()
        d = r[1]-r[0]
        for image in images:
            if not iChannel in image.channels or iChannel=='rgba':
                results.append(image)
                continue

            normalized = (image.channels[ iChannel ]-r[0])/d
            result = image.copy()
            result.channels["rgba"] = cmap(normalized, bytes=True)
            results.append(result)

        self.outputs.images.set(results)

        return 1
