from pycinema import Filter, isNumber
import logging as log

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
        images_sorted = False
        if colname:
          images = self.inputs.images.get() 

          # if the column name is in the metadata
          if colname in images[0].meta:
            log.debug("sorting on colname '" + colname + "'")
            # sort, based on the type of the data 
            if isNumber(images[0].meta[colname]):
              self.outputs.images.set(sorted(images, key= lambda k: float(k.meta[colname]), reverse=self.inputs.reverse.get()))
              images_sorted = True
            else:
              self.outputs.images.set(sorted(images, key= lambda k: str(k.meta[colname]), reverse=self.inputs.reverse.get()))
              images_sorted = True

          else:
            log.warning("colname '" + colname + "' does not exist")

        else:
          log.warning("colname '" + colname + "' is empty")

        if not images_sorted:
          results = []

          # default behavior
          for i in self.inputs.images.get():
            results.append(i.copy())

          self.outputs.images.set( results ) 

        return 1
