from pycinema import Filter

import ipywidgets
import matplotlib.colors as colors

class ColorMappingWidgets(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'container': None,
          },
          outputs={
            'map': 'plasma',
            'nan': (0,0,0,0),
            'range': (0,1),
            'channel': 'depth'
          }
        )

        self.mute = False

        def change_observer(change):
            if not self.mute and change['type'] == 'change' and change['name'] == 'value':
                self.update()
        def rgba_observer(change):
            if change['type'] == 'change' and change['name'] == 'value':
                disabled = change['new'] == 'rgba'
                self.mapWidget.disabled = disabled
                self.minWidget.disabled = disabled
                self.maxWidget.disabled = disabled
        def fixed_observer(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.fixedColorWidget.disabled = change['new'] != 'fixed'

        self.mapWidget = ipywidgets.Dropdown(
            description='Colormap:',
            options=[
                'plasma',

                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',

                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',

                'fixed'
            ],
            value='plasma'
        )
        self.mapWidget.observe(change_observer)
        self.mapWidget.observe(fixed_observer)

        self.minWidget = ipywidgets.FloatText(
            value=0,
            step=0.05,
            description='Min:'
        )
        self.minWidget.observe(change_observer)
        self.maxWidget = ipywidgets.FloatText(
            value=1,
            step=0.05,
            description='Max:'
        )
        self.maxWidget.observe(change_observer)

        self.fixedColorWidget = ipywidgets.ColorPicker(
            concise=False,
            description='Fixed Color',
            value='#f00',
            disabled=True
        )
        self.fixedColorWidget.observe(change_observer)

        self.nanColorWidget = ipywidgets.ColorPicker(
            concise=False,
            description='Nan Color',
            value='#fff'
        )
        self.nanColorWidget.observe(change_observer)

        self.channelWidget = ipywidgets.Dropdown(
            description='Channel:'
        )
        self.channelWidget.observe(change_observer)
        self.channelWidget.observe(rgba_observer)

    def _update(self):

        images = self.inputs.images.get()

        # update channels if necessary
        if len(images):
            channels = list(images[0].channels.keys())
            channels.sort()
            channels = tuple(channels)
            if self.channelWidget.options!=channels:
                self.mute = True
                self.channelWidget.options = channels
                self.mute = False
                if 'rgba' in channels:
                    self.channelWidget.value = 'rgba'
                elif 'depth' in channels:
                    self.channelWidget.value = 'depth'
                else:
                    self.channelWidget.value = channels[0]

        # add widgets to container
        container = self.inputs.container.get()
        if container!=None and len(container.children)==0:
            container.children = [
              self.channelWidget,
              self.mapWidget,
              self.minWidget,
              self.maxWidget,
              self.nanColorWidget,
              self.fixedColorWidget
            ]

        # sync outputs with widgets
        if self.mapWidget.value == 'fixed':
            self.outputs.map.set(colors.to_rgba(self.fixedColorWidget.value))
        else:
            self.outputs.map.set(self.mapWidget.value)

        self.outputs.channel.set(self.channelWidget.value)
        self.outputs.nan.set(colors.to_rgba(self.nanColorWidget.value))

        range = self.outputs.range.get()
        self.outputs.range.set((self.minWidget.value,self.maxWidget.value))

        return 1
