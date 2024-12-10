from pycinema import Filter, CinemaDatabase

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

        self.database = CinemaDatabase()

    def _update(self):

        dbPath = self.inputs.path.get()
        self.database.updatePath(dbPath)

        if not self.database.valid:
            self.outputs.table.set([[]])
            return 0

        self.outputs.table.set(self.database.table)

        return 1
