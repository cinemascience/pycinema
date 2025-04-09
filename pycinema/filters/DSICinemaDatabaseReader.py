from pycinema import Filter
import os
import re

class DSICinemaDatabaseReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': '',
            'file_column' : 'FILE',
          },
          outputs={
            'table': [[]]
          }
        )

        self.tablename = 'datacsv'
        self.dbname = 'data.db'

    def _update(self):

        from dsi.core import Terminal

        a=Terminal()

        dbPath = self.inputs.path.get()
        dbPath = os.path.expanduser(dbPath)
        dbPath = os.path.join(dbPath, self.dbname)

        a.load_module('backend','Sqlite','back-read', filename=dbPath)
        a.transload()

        # get and add column names
        table = []
        cnames = a.artifact_handler(interaction_type='get', query = 'PRAGMA table_info(' + self.tablename + ');')
        table.append([i[1] for i in cnames])

        # get and add rows
        querytable = a.artifact_handler(interaction_type='get', query = "SELECT * FROM " + self.tablename + ";")
        for r in querytable:
            data = [str(item) for item in r]
            table.append(data)

        self.outputs.table.set(table)

        return 1
