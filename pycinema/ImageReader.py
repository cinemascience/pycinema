from .Core import *

import PIL
import numpy
import h5py
import os

class ImageReader(Filter):

    def __init__(self):
        super().__init__()
        self.addInputPort("table", [])
        self.addInputPort("file_column", "FILE")
        self.addInputPort("cache", True)
        self.addOutputPort("images", [])

        self.cache = {}

    def _update(self):

        table = self.inputs.table.get()
        fileColumn = self.inputs.file_column.get()

        try:
            fileColumnIdx = list(map(str.lower,table[0])).index(fileColumn.lower())
        except ValueError as e:
            print("table does not contain '" + fileColumn + "' column!")
            return 0

        images = [];
        cache = self.inputs.cache.get()

        for i in range(1, len(table)):
            row = table[i]
            path = row[fileColumnIdx]

            filename, extension = os.path.splitext(path)
            extension = str.lower(extension[1:])

            image = None
            if cache and path in self.cache:
                images.append(self.cache[path])
                continue

            if extension == 'h5':
                image = Image()
                file = h5py.File(path, 'r')
                for (g,v) in [('channels',image.channels), ('meta',image.meta)]:
                    group = file.get(g)
                    if group==None:
                        raise ValueError('h5 file not formatted correctly')
                    for k in group.keys():
                        data = numpy.array(group.get(k))
                        if data.dtype == '|S10' and len(data)==1:
                            data = data[0].decode('UTF-8')
                        v[k.lower()] = data
                file.close()

            elif str.lower(extension) in ['png','jpg','jpeg']:
                rawImage = PIL.Image.open(path)
                if rawImage.mode == 'RGB':
                    rawImage = rawImage.convert('RGBA')
                image = Image({ 'rgba': numpy.asarray(rawImage) })

            else:
                raise ValueError('Unable to read image: '+path)

            # add meta data from data.csv
            for j in range(0, len(row)):
                key = table[0][j]
                image.meta[key] = row[j]

            if cache:
                self.cache[path] = image

            images.append( image )

        self.outputs.images.set(images)

        return 1
