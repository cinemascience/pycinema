from pycinema import Filter
import os

class DSIReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': '',
            'tablename' : 'wfdata'
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

        from dsi.core import Terminal

        a=Terminal()

        dbPath = self.inputs.path.get()
        dbPath = os.path.expanduser(dbPath)

        a.load_module('backend','Sqlite','back-read', filename=dbPath)
        a.transload()

        # get and add column names
        table = []
        cnames = a.artifact_handler(interaction_type='get', query = 'PRAGMA table_info(' + self.inputs.tablename.get() + ');')
        table.append([i[1] for i in cnames])

        # get and add rows
        querytable = a.artifact_handler(interaction_type='get', query = "SELECT * FROM " + self.inputs.tablename.get() + ";")
        for r in querytable:
            data = [str(item) for item in r]
            table.append(data)

        self.outputs.table.set(table)

        return 1
