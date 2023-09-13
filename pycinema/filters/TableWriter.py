from pycinema import Filter

import numpy
import h5py
import os
import csv
import hashlib

class TableWriter(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': '',
            'table': []
          }
        );

    def writeCSV(self):
        with open(self.inputs.path.get(), 'w') as csvfile:

            for line in self.inputs.table.get():
                csvfile.write(','.join(map(str,line)))
                csvfile.write('\n')

            csvfile.close()

    def _update(self):

        path = self.inputs.path.get()

        # get path components 
        directory, filename = os.path.split(path)

        # ensure folder exists
        if not directory == '':
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.writeCSV()


        return 1
