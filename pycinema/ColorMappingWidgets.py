from .Core import *

import ipywidgets

class ColorMappingWidgets(Filter):

    def __init__(self):
        super().__init__()
        self.addInputPort("images", [])
        self.addInputPort("container", None)

        self.addOutputPort("map", "plasma")
        self.addOutputPort("nan", (0,0,0,0))
        self.addOutputPort("range", (0,1))
        self.addOutputPort("channel", "depth")

        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.update()
        def rgba_observer(change):
            if change['type'] == 'change' and change['name'] == 'value':
                disabled = change['new'] == 'rgba'
                self.mapWidget.disabled = disabled
                self.minWidget.disabled = disabled
                self.maxWidget.disabled = disabled

        self.mapWidget = ipywidgets.Dropdown(
            description='Colormap:',
            options=[
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'plasma'
            ]
        )
        self.mapWidget.observe(on_change)

        self.minWidget = ipywidgets.FloatText(
            value=0,
            step=0.05,
            description='Min:'
        )
        self.minWidget.observe(on_change)
        self.maxWidget = ipywidgets.FloatText(
            value=1,
            step=0.05,
            description='Max:'
        )
        self.maxWidget.observe(on_change)

        self.channelWidget = ipywidgets.Dropdown(
            description='Channel:'
        )
        self.channelWidget.observe(on_change)
        self.channelWidget.observe(rgba_observer)

    def update(self):

        images = self.inputs.images.get()

        # update channels if necessary
        if len(images):
            firstImage = images[0]
            if len(self.channelWidget.options)<1:
                channels = list(firstImage.channels.keys())
                channels.sort()
                self.channelWidget.options = channels
                if 'rgba' in channels:
                    self.channelWidget.value = 'rgba'
                else:
                    self.channelWidget.value = channels[0]

        # add widgets to container
        container = self.inputs.container.get()
        if container!=None and len(container.children)==0:
            container.children = [
              self.channelWidget,
              self.mapWidget,
              self.minWidget,
              self.maxWidget
            ]

        # sync outputs with widgets
        if self.outputs.map.get() != self.mapWidget.value:
            self.outputs.map.set(self.mapWidget.value)
        if self.outputs.channel.get() != self.channelWidget.value:
            self.outputs.channel.set(self.channelWidget.value)
        range = self.outputs.range.get()
        if range[0] != self.minWidget.value or range[1] != self.maxWidget.value:
            self.outputs.range.set((self.minWidget.value,self.maxWidget.value))

        return 1
