from pycinema import Filter, isNumber

class ImageSort(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'sortBy': "",
            'reverse': 0
          },
          outputs={
            'images': []
          }
        )

    def _update(self):

        images = self.inputs.images.get()

        results = []
        if len(images)<1:
            self.outputs.images.set(results)
            return 1

        colname = self.inputs.sortBy.get()
        if colname:
          images = self.inputs.images.get() 

          # sort, based on the type of the data 
          if isNumber(images[0].meta[colname]):
            print("sorting by number")
            self.outputs.images.set(sorted(images, key= lambda k: float(k.meta[colname]), reverse=self.inputs.reverse.get()))
          else:
            print("sorting by string")
            self.outputs.images.set(sorted(images, key= lambda k: str(k.meta[colname]), reverse=self.inputs.reverse.get()))

        return 1
