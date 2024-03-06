from pycinema import Filter

import numpy
import matplotlib.cm as cm
import matplotlib.pyplot as pp

try:
  from PySide6 import QtGui, QtCore, QtWidgets
except ImportError:
  pass

class ColorMapping(Filter):

    def __init__(self):
        self.widgets = []
        self.channel_model = None
        self.maps_model = None

        super().__init__(
          inputs={
            'map': 'plasma',
            'nan': (1,1,1,1),
            'range': (0,1),
            'channel': 'depth',
            'images': [],
            'composition_id': -1
          },
          outputs={
            'images': []
          }
        )

    def updateWidgets(self):
      images = self.inputs.images.get()

      for widgets in self.widgets:
        widgets['c'].setEnabled(len(images)>0 and len(images[0].channels)>0)

      if len(images)<1:
        return
      else:
        iChannel = self.inputs.channel.get()
        iMap = self.inputs.map.get()
        iRange = self.inputs.range.get()
        iNAN = self.inputs.nan.get()
        channels = [c for c in images[0].channels]
        if self.channel_model.stringList()!=channels:
          self.channel_model.setStringList(channels)
        for widgets in self.widgets:
          widgets['c'].setCurrentIndex(channels.index(iChannel))
          widgets['m'].setCurrentIndex(self.maps_model.stringList().index(iMap))
          widgets['r'][0].setText(str(iRange[0]))
          widgets['r'][1].setText(str(iRange[1]))
          widgets['nan'].setText(str(iNAN))

    def generateWidgets(self):
        if not self.channel_model or not self.maps_model:
          self.channel_model = QtCore.QStringListModel()
          self.maps_model = QtCore.QStringListModel()
          maps = [map for map in pp.colormaps() if not map.endswith('_r')]
          maps.sort(key=lambda x: x.lower())
          self.maps_model.setStringList(maps)

        widgets = QtWidgets.QFrame()
        l = QtWidgets.QGridLayout()
        l.setAlignment(QtCore.Qt.AlignTop)
        l.setSpacing(0)
        l.setContentsMargins(0,0,0,0)
        widgets.setLayout(l)

        # Channel
        wc = QtWidgets.QComboBox()
        wc.channels = []
        l.addWidget(QtWidgets.QLabel('Channel'),0,0)
        l.addWidget(wc,0,1)
        wc.setModel(self.channel_model)

        # Color Map
        wm = QtWidgets.QComboBox()
        l.addWidget(QtWidgets.QLabel("Color Map"),1,0)
        l.addWidget(wm,1,1)
        wm.setModel(self.maps_model)

        # Range
        f = QtWidgets.QFrame()
        f.setLayout( QtWidgets.QHBoxLayout())
        l.addWidget(QtWidgets.QLabel("Range"),2,0)
        l.addWidget(f,2,1)

        wr0 = QtWidgets.QLineEdit()
        wr1 = QtWidgets.QLineEdit()
        wr0.setText('0')
        wr1.setText('1')
        r_lambda = lambda: self.inputs.range.set( (float(wr0.text()), float(wr1.text())) )
        f.layout().addWidget(wr0)
        f.layout().addWidget(wr1)

        # NAN COLOR
        wnan = QtWidgets.QPushButton()
        def select_color():
          clrpick = QtWidgets.QColorDialog()
          clrpick.setOption(QtWidgets.QColorDialog.DontUseNativeDialog)
          color = clrpick.getColor()
          if not color.isValid(): return
          color = (wnan.format(color.redF()),wnan.format(color.greenF()),wnan.format(color.blueF()),wnan.format(color.alphaF()))
          self.inputs.nan.set(color)

        wnan.clicked.connect(select_color)
        wnan.format = lambda v: float("{:.3f}".format(v))
        l.addWidget(QtWidgets.QLabel("NAN Color"),3,0)
        l.addWidget(wnan,3,1)

        self.widgets.append({
          'c': wc,
          'm': wm,
          'r': [wr0,wr1],
          'nan': wnan,
        })
        self.updateWidgets()

        # change listeners
        wc.currentIndexChanged.connect( lambda i: self.inputs.channel.set(self.channel_model.stringList()[i]) )
        wm.currentIndexChanged.connect( lambda i: self.inputs.map.set(self.maps_model.stringList()[i]) )
        wr0.editingFinished.connect(r_lambda)
        wr1.editingFinished.connect(r_lambda)

        return widgets

    def _update(self):
        images = self.inputs.images.get()
        if len(images)<1:
          self.outputs.images.set([])
          return 1
        iChannel = self.inputs.channel.get()
        channels = images[0].channels
        if iChannel not in channels:
          if 'rgba' in channels:
            self.inputs.channel.set('rgba')
          else:
            self.inputs.channel.set(channels[0])
          iChannel = self.inputs.channel.get()

        results = []
        map = self.inputs.map.get()
        nan = self.inputs.nan.get()
        composition_id = self.inputs.composition_id.get()

        nanColor = numpy.array(tuple([f * 255 for f in nan]),dtype=numpy.uint8)

        if isinstance(map, tuple):
            fixedColor = numpy.array(tuple([f * 255 for f in map]),dtype=numpy.uint8)
            for image in images:
                if not iChannel in image.channels or iChannel=='rgba':
                      results.append(image)
                      continue
                result = image.copy()
                if composition_id>=0 and 'composition_mask' in result.channels:
                    rgba = None
                    if 'rgba' not in result.channels:
                        rgba = numpy.full((result.shape[0],result.shape[1],4), nanColor, dtype=numpy.uint8)
                        result.channels['rgba'] = rgba
                    else:
                        rgba = result.channels['rgba']
                    mask0 = result.channels['composition_mask']==composition_id
                    mask1 = None
                    if iChannel == 'depth':
                        mask1 = result.channels[iChannel]==1
                    else:
                        mask1 = numpy.isnan(result.channels[iChannel])
                    rgba[mask0 & mask1] = nanColor
                    rgba[mask0 & ~mask1] = fixedColor
                else:
                    rgba = numpy.full((image.shape[0],image.shape[1],4), fixedColor, dtype=numpy.uint8)
                    mask1 = None
                    if iChannel == 'depth':
                        mask1 = result.channels[iChannel]==1
                    else:
                        mask1 = numpy.isnan(result.channels[iChannel])
                    rgba[mask1] = nanColor
                    result.channels['rgba'] = rgba

                results.append(result)
        else:
            cmap = cm.get_cmap( map )
            cmap.set_bad(color=nan )
            r = self.inputs.range.get()
            d = r[1]-r[0]
            for image in images:
                if not iChannel in image.channels or iChannel=='rgba':
                    results.append(image)
                    continue

                normalized = (image.channels[ iChannel ]-r[0])/d
                if iChannel == 'depth':
                    normalized[image.channels[iChannel]==1] = numpy.nan

                result = image.copy()
                if composition_id>=0 and 'composition_mask' in result.channels:
                    rgba = None
                    if 'rgba' not in result.channels:
                        rgba = numpy.full((result.shape[0],result.shape[1],4), nanColor, dtype=numpy.uint8)
                        result.channels['rgba'] = rgba
                    else:
                        rgba = result.channels['rgba']

                    mask = result.channels['composition_mask']==composition_id
                    rgba[mask] = cmap(normalized[mask], bytes=True)
                else:
                    result.channels["rgba"] = cmap(normalized, bytes=True)

                results.append(result)

        self.outputs.images.set(results)

        self.updateWidgets()

        return 1
