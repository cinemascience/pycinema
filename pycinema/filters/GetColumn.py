from pycinema import getTableExtent, getColumnFromTable, Filter

class GetColumn(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'table': [],
            'column': '0',
            'cast': True,
          },
          outputs={
            'array': []
          }
        )

    def _update(self):

      table = self.inputs.table.get()
      tableExtent = getTableExtent(table)
      if tableExtent[0]<1 or tableExtent[1]<1:
        self.outputs.array.set(None)
        return 1

      label = self.inputs.column.get()
      column = getColumnFromTable(
        table,
        label,
        autocast=self.inputs.cast.get()
      )
      if column:
        column.name = label

      self.outputs.array.set(column)

      return 1
