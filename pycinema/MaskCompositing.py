from .Core import *
import numpy

class MaskCompositing(Filter):

    def __init__(self):
        super().__init__()

        self.addInputPort('images_a', [])
        self.addInputPort('images_b', [])
        self.addInputPort('masks', [])
        self.addInputPort('color_channel', 'rgba')
        self.addInputPort('mask_channel', 'mask')
        self.addInputPort('opacity', 1.0)
        self.addOutputPort('images', [])

    def update(self):

        imagesA = self.inputs.images_a.get()
        imagesB = self.inputs.images_b.get()
        masks = self.inputs.masks.get()

        if not type(imagesA) is list:
          imagesA = [imagesA]
        if not type(imagesB) is list:
          imagesB = [imagesB]

        nImages = max(len(imagesA),len(imagesB))

        results = []

        colorChannel = self.inputs.color_channel.get()
        maskChannel = self.inputs.mask_channel.get()

        for i in range(0,nImages):
            A = imagesA[min(i,len(imagesA)-1)]
            B = imagesB[min(i,len(imagesB)-1)]
            M = masks[min(i,len(masks)-1)]

            if type(A) is tuple and type(B) is tuple:
                print('ERROR', 'Unable to composit just two color inputs')
                return 0

            result = None

            Ac = None
            Bc = None
            Mc = M.channels[maskChannel]

            if type(A) is tuple:
                result = B.copy()
                Bc = B.channels[colorChannel]
                Ac = numpy.full(Bc.shape,A)
            elif type(B) is tuple:
                result = A.copy()
                Ac = A.channels[colorChannel]
                Bc = numpy.full(Ac.shape,B)
            else:
                result = A.copy()
                Ac = A.channels[colorChannel]
                Bc = B.channels[colorChannel]

            mask = Mc
            if numpy.isnan(mask).any():
              mask = numpy.nan_to_num(mask,nan=0,copy=True)

            if len(Ac.shape)>2 and Ac.shape[2]>1:
              mask = numpy.stack((mask,) * Ac.shape[2], axis=-1)

            if self.inputs.opacity.get() != 1.0:
                mask = mask*self.inputs.opacity.get()
            result.channels[colorChannel] = (1-mask)*Ac + mask*Bc
            if result.channels[colorChannel].dtype != Ac.dtype:
              result.channels[colorChannel] = result.channels[colorChannel].astype(Ac.dtype)

            results.append( result )

        self.outputs.images.set(results)

        return 1
