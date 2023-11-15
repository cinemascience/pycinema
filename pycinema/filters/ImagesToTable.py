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

    def imagesToTable(images):
      if len(images)<1:
        return [[]]

      image0 = images[0]
      header = [h for h in image0.meta]

      table = []
      table.append(header)
      for image in images:
        table.append([
          str(image.meta[k]) for k in header
        ])

      return table

    def _update(self):

      images = self.inputs.images.get()

      table = ImagesToTable.imagesToTable(images)

      self.outputs.table.set(table)

      return 1;
