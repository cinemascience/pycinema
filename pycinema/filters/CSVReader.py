from pycinema import Filter, isURL
import csv
import requests
import logging as log
from os.path import exists

class CSVReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': ''
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

        table = []
        csvPath = self.inputs.path.get()

        if isURL(csvPath):
            with requests.Session() as s:
                log.info("requesting " + csvPath)
                download   = s.get(csvPath)
                decoded    = download.content.decode('utf-8')
                csvdecoded = csv.reader(decoded.splitlines(), delimiter=',')
                rows = list(csvdecoded)
                for row in rows:
                    table.append(row)

        else:
            if not csvPath:
                self.outputs.table.set([[]])
                return 0

            if not exists(csvPath):
                log.error("file not found: '" + csvPath + "'")
                self.outputs.table.set([[]])
                return 0

            try:
                with open(csvPath, 'r+') as csvfile:
                    rows = csv.reader(csvfile, delimiter=',')
                    for row in rows:
                        table.append(row)
            except:
                log.error("Unable to open file: '" + csvPath + "'")
                self.outputs.table.set([[]])
                return 0

        # remove empty lines
        table = list(filter(lambda row: len(row)>0, table))

        self.outputs.table.set(table)

        return 1
