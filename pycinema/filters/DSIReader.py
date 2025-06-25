from pycinema import Filter
import os
import re

class DSIReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': '',
            'tablename' : 'input'
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

        from dsi.core import Terminal
        from dsi.dsi  import DSI

        a=Terminal()

        dbPath = self.inputs.path.get()
        dbPath = os.path.expanduser(dbPath)
        dbTable = self.inputs.tablename.get()

        table = []
        if dbPath and dbTable:
            a.load_module('backend','Sqlite','back-read', filename=dbPath)
            a.transload()

            # get and add column names
            cnames = a.artifact_handler(interaction_type='get', query = 'PRAGMA table_info(' + self.inputs.tablename.get() + ');')
            names = cnames.values.tolist()
            colnames = []
            for i in names:
                colnames.append(i[1])
            table.append(colnames)

            # get and add rows
            if dbPath:
                if dbTable:
                    read_dsi = DSI(dbPath)
                    dsi_table = read_dsi.get_table(dbTable, True)
                    for item in dsi_table.values.tolist():
                        table.append(item[1:])

            # add id column, per pycinema expectations
            if 'id' not in table[0]:
                table[0].append('id')
                for i in range(1,len(table)):
                    table[i].append(i-1)

        self.outputs.table.set(table)

        return 1
