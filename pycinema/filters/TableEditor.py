from pycinema import Filter

import csv

class TableEditor(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'csv': '',
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

      csv_string = self.inputs.csv.get()
      lines = csv_string.split(';')
      reader = csv.reader(lines, delimiter=',')
      output = []
      for row in reader:
        output.append(row)

      self.outputs.table.set(output)

      return 1;
