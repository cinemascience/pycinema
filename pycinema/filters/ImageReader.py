from pycinema import Filter, Image

import PIL
import numpy
import h5py
import os
import re

from pycinema import getTableExtent

class ImageReader(Filter):

    def __init__(self):
        self.cache = {}

        super().__init__(
          inputs={
            'table': [[]],
            'file_column': 'FILE',
            'cache': True
          },
          outputs={
            'images': []
          }
        )

    def _update(self):

        table = self.inputs.table.get()
        tableExtent = getTableExtent(table)
        if tableExtent[0]<1 or tableExtent[1]<1:
            return self.outputs.images.set([])

        fileColumn = self.inputs.file_column.get()

        try:
            fileColumnIdx = [i for i, item in enumerate(table[0]) if re.search(fileColumn, item, re.IGNORECASE)].pop()
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
                        # print('xxx',data)
                        if data.dtype == '|S10' and len(data)==1:
                            data = data[0].decode('UTF-8')
                        # elif len(data)==1:
                        #     data = data[0]
                        v[k] = data
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
