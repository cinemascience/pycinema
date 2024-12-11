from pycinema import Filter, TableReaderObject

class TableReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': '',
            'file_column': 'FILE'
          },
          outputs={
            'table': [[]],
            'type' : ''
          }
        )

        self.tablereader = TableReaderObject()

    def _update(self):

        dbPath = self.inputs.path.get()
        self.tablereader.updatePath(dbPath)

        if not self.tablereader.valid:
            self.outputs.table.set([[]])
            return 0

        self.outputs.table.set(self.tablereader.table)
        self.outputs.type.set(self.tablereader.type)

        return 1

    def istype(self, ttype):
        return self.tablereader.type == ttype
