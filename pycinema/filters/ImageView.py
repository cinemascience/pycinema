from pycinema import Filter

import numpy
import logging as log

try:
  from PySide6 import QtGui, QtCore, QtWidgets
except ImportError:
  pass

try:
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

      self.highlight = False

    def paint(self, painter, option, widget=None):
      super().paint(painter, option, widget)
      if self.highlight:
        pen = QtGui.QPen(QtGui.QColor("#00D6E0"))
        pen.setWidth(30)
        # pen = QtGui.QPen(QtGui.QColor("#FF0000"))
        # pen.setWidth(4)
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

      self.filter.inputs.selection.set(indices,True,True)

  class _ImageViewer(QtWidgets.QGraphicsView):

      def __init__(self,filter):
          super().__init__()

          self.filter = filter

          self.mode = 0
          self.mouse_data = []

          self.setRenderHints(QtGui.QPainter.Antialiasing)
          self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
          self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
          self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
          self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
          self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
          self.setScene(filter.scene)

          self.selection_rect = QtWidgets.QGraphicsRectItem(-100, -100, 200, 200)
          self.selection_rect.setZValue(1000)
          self.selection_rect.setBrush(QtGui.QColor(0, 0, 0, 128))
          self.selection_rect.hide()
          self.scene().addItem(self.selection_rect)

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

      def update_selection_rect(self):
        w = self.mouse_data[1].x()-self.mouse_data[0].x()
        h = self.mouse_data[1].y()-self.mouse_data[0].y()
        self.selection_rect.setRect(
          self.mouse_data[0].x(),
          self.mouse_data[0].y(),
          1 if w==0 else w,
          1 if h==0 else h,
        )

        indices = [i.idx for i in self.scene().items(self.selection_rect.rect()) if isinstance(i,QtWidgets.QGraphicsPixmapItem)]
        indices.sort()

        if indices==self.filter.inputs.selection.get():
          return

        self.filter.inputs.selection.set(indices,True,True)

      def mousePressEvent(self,event):
        if event.modifiers() == QtCore.Qt.ShiftModifier:
          self.mouse_data = [
            self.mapToScene(event.pos()),
            self.mapToScene(event.pos())
          ]
          self.mode = 1
          self.selection_rect.show()
          self.update_selection_rect()
        else:
          self.mode = 0
          self.selection_rect.hide()
          super().mousePressEvent(event)

      def mouseMoveEvent(self,event):
        if self.mode == 1:
          self.mouse_data[1] = self.mapToScene(event.pos())
          self.update_selection_rect()
        else:
          super().mouseMoveEvent(event)

      def mouseReleaseEvent(self,event):
        if self.mode != 1:
          super().mouseReleaseEvent(event)
        self.mode = 0
        self.selection_rect.hide()

      def resizeEvent(self, event):
          super().resizeEvent(event)
          self.fitInView()

except NameError:
  pass

class ImageView(Filter):

    def __init__(self):
        self.scene = QtWidgets.QGraphicsScene()
        self.requiresFit = True
        self.widgets = []
        self.time_images = -2
        self.image_items = []

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

        self.image_items = []
        for i, image in enumerate(images):
          r = numpy.floor(i/nCols)
          c = i-r*nCols
          rgba = image.getChannel('rgba')
          qimage = _QGraphicsPixmapItem(rgba,i,self)
          qimage.setPos(c*max_w,r*max_h)
          self.scene.addItem(qimage)
          self.image_items.append(qimage)

    def generateWidgets(self):
        widget = _ImageViewer(self)
        self.widgets.append(widget)
        return widget

    def _update(self):
        images = self.inputs.images.get()
        nImages = len(images)

        # update images if necessary
        if self.time_images!=self.inputs.images.getTime():
          self.time_images = self.inputs.images.getTime()

          for i in self.image_items:
            self.scene.removeItem(i)

          if nImages > 0:
            self.addImages( images )
          else:
            log.warning(" no images to lay out.")

        # update selection
        for i in self.image_items:
          i.highlight = False
        selection = self.inputs.selection.get()
        for i in selection:
          if i>=0 and i<nImages:
            self.image_items[i].highlight = True
        self.outputs.images.set([ images[i] for i in selection if i>=0 and i<nImages ])

        self.scene.update()

        if self.requiresFit:
          for w in self.widgets:
            w.fitInView()
          self.requiresFit = False

        return 1
