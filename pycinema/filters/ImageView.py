from pycinema import Filter

import numpy
from PySide6 import QtCore, QtWidgets, QtGui
import logging as log

class _QGraphicsPixmapItem(QtWidgets.QGraphicsPixmapItem):
  def __init__(self,rgba,idx,filter):
    super().__init__()

    self.filter = filter
    self.idx = idx

    qimage = QtGui.QImage(
      rgba,
      rgba.shape[1], rgba.shape[0],
      rgba.shape[1] * 4,
      QtGui.QImage.Format_RGBA8888
    )
    self.setPixmap( QtGui.QPixmap(qimage) )
    self.setShapeMode(QtWidgets.QGraphicsPixmapItem.BoundingRectShape)
    self.setTransformationMode(QtCore.Qt.SmoothTransformation)

    self.highlight = self.idx in self.filter.inputs.selection.get()

  def paint(self, painter, option, widget=None):
    super().paint(painter, option, widget)
    if self.highlight:
      pen = QtGui.QPen(QtGui.QColor("#FF0000"))
      pen.setWidth(4)
      painter.setPen(pen)
      painter.drawRect(self.boundingRect())

  def mouseDoubleClickEvent(self,event):
    indices = []
    if event.modifiers() == QtCore.Qt.ControlModifier:
      indices = list(self.filter.inputs.selection.get())

    if self.idx in indices:
      indices.remove(self.idx)
    else:
      indices.append(self.idx)
    indices.sort()

    if self.filter.inputs.selection.valueIsPort():
      self.filter.inputs.selection._value.parent.inputs.value.set(indices)
    else:
      self.filter.inputs.selection.set(indices)

class _ImageViewer(QtWidgets.QGraphicsView):

    def __init__(self,scene):
        super().__init__()

        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setScene(scene)

    def fitInView(self):
        rect = QtCore.QRectF(self.scene().itemsBoundingRect())
        if rect.isNull():
            return
        self.resetTransform()
        super().fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        ZOOM_INCREMENT_RATIO = 0.1

        angle = event.angleDelta().y()
        factor = 1.0
        if angle > 0:
            factor += ZOOM_INCREMENT_RATIO
        else:
            factor -= ZOOM_INCREMENT_RATIO

        self.scale(factor, factor)

    def keyPressEvent(self,event):
        if event.key()==32:
            self.fitInView()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView()

class ImageView(Filter):

    def __init__(self):
        self.scene = QtWidgets.QGraphicsScene()
        self.requiresFit = True
        self.widgets = []

        Filter.__init__(
          self,
          inputs={
            'images': [],
            'selection': [],
          },
          outputs={
            'images': []
          }
        )

    def removeImages(self):
      for i in [i for i in self.scene.items()]:
        self.scene.removeItem(i)

    def addImages(self, images):
        max_w = 0
        max_h = 0
        if not type(images) is list:
          return 1

        for image in images:
          max_w = max(image.shape[1],max_w)
          max_h = max(image.shape[0],max_h)
        # margin
        max_w += 10
        max_h += 10

        n = max(len(images),1)
        nRows = int(numpy.floor(numpy.sqrt(n)))
        nCols = int(numpy.ceil(n/nRows))

        total_h = 4*nRows*max_h
        total_w = 4*nCols*max_w

        rect = QtCore.QRectF(-total_w,-total_h,2*total_w,2*total_h)
        self.requiresFit = self.scene.sceneRect() != rect
        if self.requiresFit:
          self.scene.setSceneRect(rect)

        for i, image in enumerate(images):
          r = numpy.floor(i/nCols)
          c = i-r*nCols
          rgba = image.getChannel('rgba')
          qimage = _QGraphicsPixmapItem(rgba,i,self)
          qimage.setPos(c*max_w,r*max_h)
          self.scene.addItem(qimage)

    def generateWidgets(self):
        widget = _ImageViewer(self.scene)
        self.widgets.append(widget)
        return widget

    def _update(self):
        self.removeImages()
        images = self.inputs.images.get()
        nImages = len(images)
        if nImages > 0:
          self.addImages( images )
        else:
          log.warning(" no images to lay out.")

        self.outputs.images.set([ images[i] for i in self.inputs.selection.get() if i>=0 and i<nImages ])

        if self.requiresFit:
          for w in self.widgets:
            w.fitInView()
          self.requiresFit = False

        return 1
