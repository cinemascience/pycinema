from pycinema import Filter

from .FilterView import FilterView

from PySide6 import QtCore, QtWidgets

import numpy
import matplotlib.cm as cm
import matplotlib.pyplot as pp

class ColorMappingView(Filter, FilterView):

    def __init__(self):
        FilterView.__init__(
          self,
          filter=self,
          delete_filter_on_close = True
        )

        Filter.__init__(
          self,
          inputs={
            'images': [],
            'channel': 'rgba',
            'map': 'plasma',
            'range': (0,1),
            'nan': (1,1,1,1),
            'composition_id': -1
          },
          outputs={
            'images': []
          }
        )

    def generateWidgets(self):
        self.widgetsDict = {}
        self.widgets = QtWidgets.QFrame()
        l = QtWidgets.QGridLayout()
        l.setAlignment(QtCore.Qt.AlignTop)
        l.setSpacing(0)
        l.setContentsMargins(0,0,0,0)
        self.widgets.setLayout(l)

        self.content.layout().addWidget(self.widgets)

        gridL = self.widgets.layout()
        row = 0

        w = QtWidgets.QComboBox()
        self.widgetsDict['channel'] = w
        w.channels = []
        gridL.addWidget(QtWidgets.QLabel("Channel"),row,0)
        gridL.addWidget(w,row,1)
        w.on_text_changed = lambda v: self.inputs.channel.set(v)
        w.currentTextChanged.connect( w.on_text_changed )
        row += 1

        w = QtWidgets.QComboBox()
        self.widgetsDict['map'] = w
        gridL.addWidget(QtWidgets.QLabel("Color Map"),row,0)
        gridL.addWidget(w,row,1)
        maps = [map for map in pp.colormaps() if not map.endswith('_r')]
        maps.sort(key=lambda x: x.lower())
        w.addItems(
          maps
        )
        w.currentTextChanged.connect( lambda v: self.inputs.map.set(v) )
        row += 1

        # RANGE
        f = QtWidgets.QFrame()
        f.setLayout( QtWidgets.QHBoxLayout())
        gridL.addWidget(QtWidgets.QLabel("Range"),row,0)
        gridL.addWidget(f,row,1)

        w0 = QtWidgets.QLineEdit()
        w1 = QtWidgets.QLineEdit()
        self.widgetsDict['range'] = [w0,w1]
        l = lambda v: self.inputs.range.set( (float(w0.text()), float(w1.text())) )
        w0.textEdited.connect(l)
        w1.textEdited.connect(l)
        f.layout().addWidget(w0)
        f.layout().addWidget(w1)
        row+=1

        # NAN COLOR
        def select_color():
            w = self.widgetsDict['nan']
            clrpick = QtWidgets.QColorDialog()
            clrpick.setOption(QtWidgets.QColorDialog.DontUseNativeDialog)
            color = clrpick.getColor()
            color = (w.format(color.redF()),w.format(color.greenF()),w.format(color.blueF()),w.format(color.alphaF()))
            w.setText(str(color))
            self.inputs.nan.set(color)

        w = QtWidgets.QPushButton()
        w.clicked.connect(select_color)
        w.format = lambda v: float("{:.3f}".format(v))
        self.widgetsDict['nan'] = w
        gridL.addWidget(QtWidgets.QLabel("NAN Color"),row,0)
        gridL.addWidget(w,row,1)
        row += 1

    def update_widgets(self,images):
        self.widgetsDict['map'].setCurrentText(self.inputs.map.get())
        self.widgetsDict['range'][0].setText(str(self.inputs.range.get()[0]))
        self.widgetsDict['range'][1].setText(str(self.inputs.range.get()[1]))
        self.widgetsDict['nan'].setText(str(self.inputs.nan.get()))

        # channels
        channels = set({})
        for i in images:
            for c in i.channels:
                channels.add(c)
        new_channels = list(channels)
        new_channels.sort()

        w = self.widgetsDict['channel']
        w.currentTextChanged.disconnect( w.on_text_changed )
        w.clear()
        for c in new_channels:
            w.addItem(c)
        w.channels = new_channels
        w.currentTextChanged.connect( w.on_text_changed )
        channel = self.inputs.channel.get()
        if channel in new_channels:
          w.setCurrentText( channel )
        else:
          self.inputs.channel.set(new_channels[0])
          w.setCurrentText( new_channels[0] )

    def _update(self):
        i_images = self.inputs.images.get()
        if len(i_images)<1:
          self.outputs.images.set([])
          return 1

        self.update_widgets(i_images)

        i_channel = self.inputs.channel.get()
        i_map = self.inputs.map.get()
        i_nan = self.inputs.nan.get()
        i_composition_id = self.inputs.composition_id.get()

        nanColor = numpy.array(tuple([f * 255 for f in i_nan]),dtype=numpy.uint8)

        cmap = cm.get_cmap( i_map )
        cmap.set_bad(color=i_nan )
        i_range = self.inputs.range.get()
        d = i_range[1]-i_range[0]

        o_images = []
        for i_image in i_images:
            if not i_channel in i_image.channels or i_channel=='rgba':
                o_images.append(i_image)
                continue

            normalized = (i_image.channels[ i_channel ]-i_range[0])/d
            if i_channel == 'depth':
                normalized[i_image.channels[i_channel]==1] = numpy.nan

            o_image = i_image.copy()
            if i_composition_id>=0 and 'composition_mask' in o_image.channels:
                rgba = None
                if 'rgba' not in o_image.channels:
                    rgba = numpy.full((o_image.shape[0],o_image.shape[1],4), nanColor, dtype=numpy.uint8)
                    o_image.channels['rgba'] = rgba
                else:
                    rgba = o_image.channels['rgba']

                mask = o_image.channels['composition_mask']==i_composition_id
                rgba[mask] = cmap(normalized[mask], bytes=True)
            else:
                o_image.channels["rgba"] = cmap(normalized, bytes=True)

            o_images.append(o_image)

        self.outputs.images.set(o_images)

        return 1
