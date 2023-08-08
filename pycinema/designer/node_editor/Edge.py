from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.designer.node_editor.NodeEditorStyle import *

class Edge(QtWidgets.QGraphicsLineItem):

    def __init__(self,port0,port1,parent=None):
        super().__init__(parent)

        self.setZValue(Z_EDGE_LAYER)

        self.setPen(QtGui.QPen(COLOR_NORMAL, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

        self.port0 = port0
        self.port1 = port1

        self.port0.parentItem().s_moved.connect(self.update)
        self.port1.parentItem().s_moved.connect(self.update)

        self.update()

    def update(self):
        self.setLine( QtCore.QLineF(
          self.port0.mapToItem(self, self.port0.disc.boundingRect().center()),
          self.port1.mapToItem(self, self.port1.disc.boundingRect().center())
        ))