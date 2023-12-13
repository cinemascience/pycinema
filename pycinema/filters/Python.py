from pycinema import Filter

class Python(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'inputs': None,
            'code': ''
          },
          outputs={
            'outputs': None
          }
        )

    def _update(self):

      variables = {}
      variables['inputs'] = self.inputs.inputs.get()
      variables['outputs'] = None

      exec(self.inputs.code.get(),variables)
      output = variables['outputs']

      self.outputs.outputs.set(output)

      return 1;
