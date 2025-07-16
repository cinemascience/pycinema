from pycinema import Filter
import numpy as np

class ImageMetrics(Filter):
    """!
    @brief A class that computes image metrics on float images

    """

    def __init__(self):
        """!
        @brief Constructor for ImageMetrics
        @param images A list of float images
        @param table A table resulting from the computation of metrics
        """

        super().__init__(
          inputs={
            'images': []
          },
          outputs={
            'table': []
          }
        )

    def _update(self):
        newtable = []

        channels = []
        images = self.inputs.images.get()
        for c,data in images[0].channels.items():
            channels.append(c)

        # the first two columns must be id and Timestep
        allids = ['id','Timestep']
        allids.extend(channels)
        newtable.append(allids)
        # compute the average of each channel
        for i in self.inputs.images.get():
            # make a row for each image
            newtable.append([])
            newtable[-1].append(i.meta['id'])
            newtable[-1].append(i.meta['Timestep'])
            for c in channels:
                ave = np.nanmean(i.getChannel(c))
                newtable[-1].append(ave)

        self.outputs.table.set(newtable)
