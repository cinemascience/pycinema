from pycinema import Filter, getTableExtent, isNumber

import copy
import numpy as np

class Calculator(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'table': [[]],
            'label': 'result',
            'expression': ''
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

      iTable = self.inputs.table.get()
      extent = getTableExtent(iTable)

      if extent[0]<2 or extent[1]<1:
        self.outputs.table.set([[]])
        return

      oTable = copy.deepcopy(iTable)

      variables = {}
      for c in range(0,extent[1]):
        if isNumber(oTable[1][c]):
          variables[oTable[0][c]] = np.array([float(row[c]) for row in oTable[1:]])

      variables['__result'] = ''

      expression = self.inputs.expression.get().strip()
      if expression=='':
        expression = '0'

      exec('__result = '+expression,variables)
      result = variables['__result']

      if type(result) not in [list, np.ndarray]:
        result = [result for i in range(0,extent[0]-1)]

      oTable[0].append( self.inputs.label.get() )
      for i in range(1,extent[0]):
        oTable[i] = list(oTable[i])
        oTable[i].append(float(result[i-1]))

      self.outputs.table.set(oTable)

      return 1;
