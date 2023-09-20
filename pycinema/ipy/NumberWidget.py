from pycinema import Filter

import ipywidgets

class NumberWidget(Filter):

    def __init__(self,port,range=None):
        super().__init__(
          outputs={
            'number': port.get()
          }
        )

        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.update()

        if range:
            self.widget = ipywidgets.FloatSlider(
                description=port.name,
                value=port.get(),
                min = range[0],
                max = range[1],
                step= range[2],
            )
        else:
            self.widget = ipywidgets.FloatText(
                description=port.name,
                value=port.get()
            )

        self.widget.observe(on_change)

        port.set(self.outputs.number, False)

    def _update(self):
        self.outputs.number.set(float(self.widget.value))
        return 1
