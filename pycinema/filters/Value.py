from pycinema import Filter

class Value(Filter):

    def __init__(self):
        Filter.__init__(
          self,
          inputs = {
            'value': ''
          },
          outputs = {
            'value': ''
          }
        )

    def _update(self):
        value = self.inputs.value.get()
        self.outputs.value.set( value )
        return 1
