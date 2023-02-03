from .Core import *
import numpy
import matplotlib.cm as cm

class ColorMapping(Filter):

    def __init__(self):
        super().__init__()

        self.addInputPort("map", "plasma")
        self.addInputPort("nan", (1,1,1,1))
        self.addInputPort("range", (0,1))
        self.addInputPort("channel", "depth")
        self.addInputPort("images", [])
        self.addInputPort("composition_id", -1)
        self.addOutputPort("images", [])

    def _update(self):

        images = self.inputs.images.get()
        iChannel = self.inputs.channel.get()
        results = []
        map = self.inputs.map.get()
        nan = self.inputs.nan.get()
        composition_id = self.inputs.composition_id.get()

        nanColor = numpy.array(tuple([f * 255 for f in nan]),dtype=numpy.uint8)

        if isinstance(map, tuple):
            fixedColor = numpy.array(tuple([f * 255 for f in map]),dtype=numpy.uint8)
            for image in images:
                if not iChannel in image.channels or iChannel=='rgba':
                      results.append(image)
                      continue
                result = image.copy()
                if composition_id>=0 and 'composition_mask' in result.channels:
                    rgba = None
                    if 'rgba' not in result.channels:
                        rgba = numpy.full((result.shape[0],result.shape[1],4), nanColor, dtype=numpy.uint8)
                        result.channels['rgba'] = rgba
                    else:
                        rgba = result.channels['rgba']
                    mask0 = result.channels['composition_mask']==composition_id
                    mask1 = None
                    if iChannel == 'depth':
                        mask1 = result.channels[iChannel]==1
                    else:
                        mask1 = numpy.isnan(result.channels[iChannel])
                    rgba[mask0 & mask1] = nanColor
                    rgba[mask0 & ~mask1] = fixedColor
                else:
                    rgba = numpy.full((image.shape[0],image.shape[1],4), fixedColor, dtype=numpy.uint8)
                    mask1 = None
                    if iChannel == 'depth':
                        mask1 = result.channels[iChannel]==1
                    else:
                        mask1 = numpy.isnan(result.channels[iChannel])
                    rgba[mask1] = nanColor
                    result.channels['rgba'] = rgba

                results.append(result)
        else:
            cmap = cm.get_cmap( map )
            cmap.set_bad(color=nan )
            r = self.inputs.range.get()
            d = r[1]-r[0]
            for image in images:
                if not iChannel in image.channels or iChannel=='rgba':
                    results.append(image)
                    continue

                normalized = (image.channels[ iChannel ]-r[0])/d
                if iChannel == 'depth':
                    normalized[image.channels[iChannel]==1] = numpy.nan

                result = image.copy()
                if composition_id>=0 and 'composition_mask' in result.channels:
                    rgba = None
                    if 'rgba' not in result.channels:
                        rgba = numpy.full((result.shape[0],result.shape[1],4), nanColor, dtype=numpy.uint8)
                        result.channels['rgba'] = rgba
                    else:
                        rgba = result.channels['rgba']

                    mask = result.channels['composition_mask']==composition_id
                    rgba[mask] = cmap(normalized[mask], bytes=True)
                else:
                    result.channels["rgba"] = cmap(normalized, bytes=True)

                results.append(result)

        self.outputs.images.set(results)

        return 1
