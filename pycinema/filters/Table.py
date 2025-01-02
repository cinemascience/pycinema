from pycinema import Filter, TableReaderObject

class Table(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': ''
          },
          outputs={
            'table': [[]],
            'attributes': {'table':{},'columns':{}}
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
        self.outputs.attributes.get()['table']['type'] = self.tablereader.type

        return 1

    def istype(self, ttype):
        return self.tablereader.type == ttype
