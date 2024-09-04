from pycinema import Filter

import csv
import os.path
import re
import logging as log

class CinemaDatabaseReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': '',
            'file_column': 'FILE'
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

        table = []
        dbPath = self.inputs.path.get()
        dbPath = os.path.expanduser(dbPath)

        if not dbPath:
            self.outputs.table.set([[]])
            return 0

        if not os.path.exists(dbPath):
            log.error(" CDB not found '" + dbPath + "'")
            self.outputs.table.set([[]])
            return 0

        try:
            dataCsvPath = dbPath + '/data.csv'
            with open(dataCsvPath, 'r+') as csvfile:
                rows = csv.reader(csvfile, delimiter=',')
                for row in rows:
                    table.append(row)
        except:
            log.error(" Unable to open data.csv")
            self.outputs.table.set([[]])
            return 0

        # remove empty lines
        table = list(filter(lambda row: len(row)>0, table))

        # add dbPath prefix to file columns
        fileColumnIndices = [i for i, item in enumerate(table[0]) if re.search(self.inputs.file_column.get(), item, re.IGNORECASE)]
        for i in range(1,len(table)):
          for j in fileColumnIndices:
            if not table[i][j].startswith('http:') and not table[i][j].startswith('https:'):
                table[i][j] = dbPath + '/' + table[i][j]

        # add id column
        if 'id' not in table[0]:
          table[0].append('id')
          for i in range(1,len(table)):
            table[i].append(i-1)

        self.outputs.table.set(table)

        return 1
