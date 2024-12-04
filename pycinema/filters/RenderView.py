from pycinema import Filter

import numpy
import logging as log

try:
  from PySide6 import QtGui, QtCore, QtWidgets
except ImportError:
  pass

try:
  class _QGraphicsPixmapItem(QtWidgets.QGraphicsPixmapItem):
    def __init__(self):
      super().__init__()

      self.setShapeMode(QtWidgets.QGraphicsPixmapItem.BoundingRectShape)
      self.setTransformationMode(QtCore.Qt.SmoothTransformation)

    def update(self,rgba):
      qimage = QtGui.QImage(
        rgba,
        rgba.shape[1], rgba.shape[0],
        rgba.shape[1] * 4,
        QtGui.QImage.Format_RGBA8888
      )
      self.setPixmap( QtGui.QPixmap(qimage) )

    def paint(self, painter, option, widget=None):
      super().paint(painter, option, widget)

  class _ImageViewer(QtWidgets.QGraphicsView):

      def __init__(self,filter):
          super().__init__()

          self.filter = filter

          self.mode = 0
          self.mouse_pos_0 = None
          self.camera_0 = None

          self.setRenderHints(QtGui.QPainter.Antialiasing)
          self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
          self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
          self.setScene(filter.scene)

      def fitInView(self):
        rect = QtCore.QRectF(self.scene().itemsBoundingRect())
        if rect.isNull():
          return
        self.resetTransform()
        super().fitInView(rect, QtCore.Qt.KeepAspectRatio)

      def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView()

      def wheelEvent(self, event):
        return

      def keyPressEvent(self,event):
        if event.key()==32:
          self.fitInView()

      def mousePressEvent(self,event):
        self.mode = 1
        self.mouse_pos_0 = event.pos()
        self.camera_0 = self.filter.inputs.camera.get()
        super().mousePressEvent(event)

      def mouseMoveEvent(self,event):
        if self.mode != 1: return

        delta = event.pos() - self.mouse_pos_0
        factor = 0.1
        camera_1 = [
          # round(sorted([0, 360, self.camera_0[0] + delta.x()*factor])[1], 2),
          round((self.camera_0[0] + delta.x()*factor), 2),
          round(sorted([-90, 90, self.camera_0[1] + delta.y()*factor])[1], 2),
        ]
        self.filter.inputs.camera.set(camera_1,True,True)

        super().mouseMoveEvent(event)

      def mouseReleaseEvent(self,event):
        if self.mode != 1: return

        self.mode = 0
        super().mouseReleaseEvent(event)

except NameError:
  pass

class RenderView(Filter):

    def __init__(self):
        self.scene = QtWidgets.QGraphicsScene()

        self.canvas = _QGraphicsPixmapItem()
        self.scene.addItem(self.canvas)

        self.widgets = []

        Filter.__init__(
          self,
          inputs={
            'images': [],
            'camera': [30,30] #FIXME: cause gimbal lock at 0, 0
          },
          outputs={
            'images': []
          }
        )

    def generateWidgets(self):
        widget = _ImageViewer(self)
        self.widgets.append(widget)
        return widget

    def _update(self):
        images = self.inputs.images.get()
        if len(images)<1 or 'rgba' not in images[0].channels:
          print('[WARNING:RenderView] No input image or missing rgba channel')
          self.canvas.update(numpy.zeros((1,1,4)))
          return 1

        self.canvas.update(images[0].channels['rgba'])

        for w in self.widgets:
            w.fitInView()

        return 1
