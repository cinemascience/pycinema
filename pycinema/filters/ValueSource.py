from pycinema import Filter

class ValueSource(Filter):

    def __init__(self):
        Filter.__init__(
          self,
          inputs = {
            'value': []
          },
          outputs = {
            'value': ''
          }
        )

    def _update(self):
        value = self.inputs.value.get()
        if isinstance(value, str):
          value = eval(value)
        self.outputs.value.set( value )
        return 1
