from pycinema import Filter, imageFromMatplotlibFigure

import matplotlib.pyplot as plt
# use offscreen backend
import matplotlib as mpl
mpl.use('Agg')

from itertools import groupby

class Plot(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'items': [],
            'dpi': 100,
          },
          outputs={
            'images': []
          }
        )

    def _update(self):

      items = self.inputs.items.get()
      items = items if isinstance(items,list) else [items]

      key_lambda = lambda i: i['x']['label']
      items.sort(key=key_lambda)
      grouped_items = [list(group) for key, group in groupby(items, key=key_lambda)]

      res = []
      for group in grouped_items:
        # figure = plt.figure(dpi=int(self.inputs.dpi.get()))
        figure = plt.figure()
        # plt.margins(0, x=None, y=None, tight=True)
        plt.grid(color='#ccc', linestyle='-', linewidth=1)
        figure.tight_layout()
        xLabel = group[0]['x']['label']
        xData = group[0]['x']['data']
        for i in group:
          plt.plot(
            xData,
            i['y']['data'],
            i['fmt'],
            **i['style'],
            label=i['y']['label'] if i['x']['label']=='index' else i['x']['label']+' x '+i['y']['label'],
          )

        plt.legend()
        res.append(imageFromMatplotlibFigure(figure,self.inputs.dpi.get()))

      self.outputs.images.set(res)

      return 1;
