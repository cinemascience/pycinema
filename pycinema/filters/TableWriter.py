from pycinema import Filter

import os
import csv

#
# TableWriter
#
# writes its input table as a csv file
#
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
            write = csv.writer(csvfile)
            write.writerows(self.inputs.table.get())


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
