from .FilterView import ViewFilter

import numpy
# import PIL
from PySide6 import QtCore, QtWidgets, QtGui

class _ImageViewer(QtWidgets.QGraphicsView):

    last_added_node = None
    mouse_pos0 = None
    mouse_pos1 = None

    def __init__(self):
        super().__init__()
        self._zoom = 1

        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        l = 10000
        self.setSceneRect(-l,-l,2*l,2*l)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self.items = []

    def removeImages(self):
        for item in self.items:
            self._scene.removeItem(item)
            # item.setParent(None)
        self.items = []

    def addImage(self,image):
        item = QtWidgets.QGraphicsPixmapItem()

        rgba = image.channels['rgba']
        qimage = QtGui.QImage(
          rgba,
          rgba.shape[1], rgba.shape[0],
          rgba.shape[1] * 4,
          QtGui.QImage.Format_RGBA8888
        )
        item.setPixmap( QtGui.QPixmap(qimage) )
        item.setShapeMode(QtWidgets.QGraphicsPixmapItem.BoundingRectShape)
        item.setTransformationMode(QtCore.Qt.SmoothTransformation)

        rect = self._scene.itemsBoundingRect()

        item.setPos(0,rect.bottom()+10)

        self._scene.addItem(item)
        self.items.append(item)

    def fitInView(self):
        rect = QtCore.QRectF(self._scene.itemsBoundingRect())
        if rect.isNull():
            return

        self.resetTransform()
        self._zoom = 1.0
        super().fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        ZOOM_INCREMENT_RATIO = 0.1

        angle = event.angleDelta().y()
        factor = 1.0
        if angle > 0:
            factor += ZOOM_INCREMENT_RATIO
        else:
            factor -= ZOOM_INCREMENT_RATIO

        self._zoom *= factor
        self.scale(factor, factor)

    def keyPressEvent(self,event):
        if event.key()==32:
            self.fitInView()

class ImageViewer(ViewFilter):

    def __init__(self, view):
        self.view = _ImageViewer()

        view.content.layout().addWidget(self.view,1)

        super().__init__(inputs={
            'images': []
        })

    def _update(self):
        self.view.removeImages()

        for image in self.inputs.images.get():
            self.view.addImage(image)

        self.view.fitInView()
        return 1
