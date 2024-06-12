from pycinema import Filter

import numpy
import h5py
import os
import csv
import hashlib
import re
import PIL

class CinemaDatabaseWriter(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'path': '',
            'ignore': ['^id','^camera'],
            'hdf5': True,
          }
        );

    def getImageHash(self,idx,header,image):
        text = str(idx)+';'
        for p in header:
            text = text + str(image.meta[p]) + ';'
        return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)

    def writeH5(self,path,image):
        file = h5py.File(path, 'w')

        fChannel = file.create_group('channels')
        for k in image.channels:
            fChannel.create_dataset(k,data=image.channels[k], compression='gzip', compression_opts=9)

        fMeta = file.create_group('meta')
        for k in image.meta:
            data = image.meta[k]
            if type(data) is set:
                continue

            if type(data) is numpy.ndarray and data.size>1:
                fMeta.create_dataset(k,data=data, compression='gzip', compression_opts=9)
            elif isinstance(data, str):
                fMeta.create_dataset(k,data=numpy.array([data],dtype='S10'))
            else:
                fMeta.create_dataset(k,data=data)

        file.close()

    def writePNG(self,path,image):
        rgba_array = image.channels['rgba']
        image = PIL.Image.fromarray(rgba_array, 'RGBA')
        image.save(path, 'PNG')

    def _update(self):

        images = self.inputs.images.get()
        path = self.inputs.path.get()
        path = os.path.expanduser(path)

        if len(images)<1 or len(path)<1:
            return 1

        # check if path is a cdb
        filename, extension = os.path.splitext(path)
        extension = str.lower(extension[1:])

        # ensure folder exists
        if not os.path.exists(path):
            os.makedirs(path)

        use_hdf5 = self.inputs.hdf5.get()

        if extension == 'cdb':
            # cdb folder
            csvPath = path+'/data.csv'
            csvData = []

            image0 = images[0]
            header = [p for p,v in image0.meta.items() if not type(v) is numpy.ndarray or len(v)==1]
            ignore = self.inputs.ignore.get()
            ignore.append('^FILE')
            header = [p for p in header if not any([re.search(i, p, re.IGNORECASE) for i in ignore])]
            header.sort()
            csvData.append(header + ['FILE'])

            # write images
            for i,image in enumerate(images):
                imageHash = self.getImageHash(i,header,image)
                fileName = str(imageHash)+('.h5' if use_hdf5 else '.png')
                row = [str(image.meta[p]) for p in header]
                row.append(fileName)
                csvData.append(row)

                # delete all FILE meta data entries
                out_image = image.copy()
                for m in [m for m in list(out_image.meta.keys()) if re.search('^file', m, re.IGNORECASE)]:
                    del out_image.meta[m]

                # write image
                if use_hdf5:
                  self.writeH5(path+'/'+fileName,out_image)
                else:
                  self.writePNG(path+'/'+fileName,out_image)

            # write csv
            with open(csvPath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in csvData:
                    writer.writerow(row)

        elif extension == '':
            # write image
            for i,out_image in enumerate(images):
              if use_hdf5:
                self.writeH5(path+str(i)+'.h5',out_image)
              else:
                self.writePNG(path+str(i)+'.png',out_image)

        return 1
