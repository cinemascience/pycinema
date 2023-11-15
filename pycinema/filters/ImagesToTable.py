from pycinema import Filter, isNumber

class ImagesToTable(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'images': [[]],
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

      images = self.inputs.images.get()

      if len(images)<1:
        self.outputs.table.set([[]])
        return 1

      image0 = images[0]
      header = [h for h in image0.meta]

      table = []
      table.append(header)
      for image in images:
        table.append([
          image.meta[k] for k in header
        ])

      self.outputs.table.set(table)

      return 1;
