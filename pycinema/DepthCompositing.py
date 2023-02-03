from .Core import *
import numpy

class DepthCompositing(Filter):

    def __init__(self):
        super().__init__()

        self.addInputPort('images_a', [])
        self.addInputPort('images_b', [])
        self.addInputPort('depth_channel', 'depth')
        self.addInputPort('composite_by_meta', (None,{}))
        self.addOutputPort('images', [])

    def toTuple(self,data):
        if type(data) == numpy.ndarray:
            data = data.tolist()
            if type(data) != list:
                data = tuple([data])
            else:
                data = tuple(data)
        if type(data)==tuple and len(data)==1:
            data = data[0]
        return data

    def toList(self,data):
        if type(data) == numpy.ndarray:
            data = data.tolist()
            if type(data) != list:
                return [data]
            else:
                return data
        elif type(data) == list:
            return data
        return [data]

    def getKeys(self,image,composite_by_meta):
        composite_by_meta = [composite_by_meta[0]] + ['id','file']
        return list(filter(lambda m: m not in composite_by_meta, image.meta))

    def getTupleKey(self,image,keys):
        meta_keys = [self.toList(image.meta[m]) for m in keys]
        return tuple(sum(meta_keys, []))

    def makeSet(self,data):
        if type(data)==set:
            return data

        t = self.toTuple(data)
        return set([t])

        # elif type(data) == numpy.ndarray:
        #     if data.shape==():
        #         return set([data.item()])
        #     elif data.shape==(1,):
        #         return set([data[0]])
        #     else:
        #         return set([tuple(data)])
        # elif type(data) == tuple:
        #     if len(data)==1:
        #         return set([data[0]])
        #     else:
        #         return set([tuple(data)])
        #     return set([data])
        # elif type(data) in [float,int] or numpy.isscalar(data):
        #     return set([data])
        # else:
        #     return set([tuple([data])])

    def compose(self,A,B,depthChannel):

        result = A.copy()
        mask = A.channels[depthChannel] > B.channels[depthChannel]

        for c in A.channels:
            if c not in B.channels:
                continue

            data = numpy.copy(A.channels[c])
            data[mask] = B.channels[c][mask]
            result.channels[c] = data

        for m in B.meta:
            if m not in A.meta:
                result.meta[m] = b.meta[m]

        for m in A.meta:
            if m in B.meta:
                A_as_Set = self.makeSet(A.meta[m])
                B_as_Set = self.makeSet(B.meta[m])
                union = A_as_Set.union(B_as_Set)
                if len(union)==1:
                    union = list(union)[0]
                result.meta[m] = union

        if 'composition_mask' in result.channels:
            metaCompositing = self.inputs.composite_by_meta.get()
            result.channels['composition_mask'][mask] = metaCompositing[1][str(B.meta[metaCompositing[0]])]

        return result

    def _update(self):

        imagesA = self.inputs.images_a.get()
        imagesB = self.inputs.images_b.get()

        results = []

        nImages = len(imagesA)
        if len(imagesB)>0 and nImages!=len(imagesB):
          print('ERROR', 'Input image lists must be of equal size.' )
          self.outputs.images.set(results)
          return 0

        depthChannel = self.inputs.depth_channel.get()

        if len(imagesA)==len(imagesB):
            for i in range(0,nImages):
                results.append(
                    self.compose(imagesA[i],imagesB[i],depthChannel)
                )
        elif len(imagesA)>0:
            imagesMap = {}

            metaCompositing = self.inputs.composite_by_meta.get()
            keys = self.getKeys(imagesA[0],metaCompositing)

            for i in imagesA:
                key = self.getTupleKey(i,keys)
                if not key in imagesMap:
                    imagesMap[key] = []
                imagesMap[key].append(i)

            for key, images in imagesMap.items():
                # print(len(images),key)
                result = images[0].copy()

                if metaCompositing[0]!=None and 'composition_mask' not in result.channels:
                    result.channels['composition_mask'] = numpy.full(result.shape[:2], metaCompositing[1][str(result.meta[metaCompositing[0]])], dtype=numpy.ubyte)
                for i in range(1,len(images)):
                    result = self.compose(result,images[i],depthChannel)
                results.append(result)

        self.outputs.images.set(results)

        return 1
