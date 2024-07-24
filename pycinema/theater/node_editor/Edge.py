from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.node_editor.NodeEditorStyle import NodeEditorStyle as NES

class Edge(QtWidgets.QGraphicsPathItem):

    def __init__(self,port0,port1,parent=None):
        super().__init__(parent)

        self.setZValue(NES.Z_EDGE_LAYER)


        image_port_names = ['image','images']
        is_image = port0.port.name in image_port_names or port1.port.name in image_port_names

        if is_image:
          self.setPen(QtGui.QPen(NES.COLOR_NORMAL, 2, QtCore.Qt.DashLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        elif port0.port.name=='table' or port1.port.name=='table':
          self.setPen(QtGui.QPen(NES.COLOR_NORMAL, 2, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        else:
          self.setPen(QtGui.QPen(NES.COLOR_NORMAL, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

        self.port0 = port0
        self.port1 = port1

        self.port0.parentItem().s_moved.connect(self.update)
        self.port1.parentItem().s_moved.connect(self.update)

        self.update()

    def update(self):
        p0 = self.port0.mapToItem(self, self.port0.disc.boundingRect().center())
        p1 = self.port1.mapToItem(self, self.port1.disc.boundingRect().center())

        x0 = p0.x()
        y0 = p0.y()
        x1 = p1.x()
        y1 = p1.y()
        path = QtGui.QPainterPath()
        path.moveTo(p0)
        dx = abs(x0 - x1)
        if x0<x1: dx *= 0.5
        path.cubicTo(
          x0+dx, y0,
          x1-dx, y1,
          x1,y1
        )

        self.setPath(path)

