from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.designer.node_editor.NodeEditorStyle import *
from pycinema.designer.node_editor.Port import Port

class Node(QtWidgets.QGraphicsObject):

    node_map = {}

    s_moved = QtCore.Signal(name='s_moved')

    def __init__(self,filter):
        super().__init__()

        self.setZValue(Z_NODE_LAYER)

        self.node_map[filter] = self

        self.filter = filter
        self.inputPorts = self.filter.inputs.ports()
        self.outputPorts = self.filter.outputs.ports()

        brN = self.boundingRect()

        self.setFlags(
              QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )

        # label
        self.label = QtWidgets.QGraphicsProxyWidget(self)
        label_ = QtWidgets.QLabel(self.filter.id)
        font = QtGui.QFont()
        # font.setBold(True)
        font.setPointSize(10)
        label_.setFont(font)
        label_.setStyleSheet("background-color: transparent; color: "+COLOR_NORMAL_);
        self.label.setWidget(label_)

        brL = self.label.boundingRect()

        nodeWidthToLabelWidthRatio = brN.width() / brL.width();
        if nodeWidthToLabelWidthRatio < 1.0:
            self.label.setScale(nodeWidthToLabelWidthRatio);
        else:
            nodeWidthToLabelWidthRatio = 1.0

        self.label.setPos(
          brN.center().x()-brL.width()*nodeWidthToLabelWidthRatio/2,
          NODE_HEADER_HEIGHT/2 - brL.height()/2
        )

        # output ports
        portY = NODE_PORT_Y

        for name, port in self.outputPorts:
            qPort = Port(self,port)
            qPort.setPos(brN.right(),portY)
            portY += NODE_PORT_SPACE

        for name, port in self.inputPorts:
            qPort = Port(self,port)
            qPort.setPos(brN.left(),portY)
            portY += NODE_PORT_SPACE


    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.s_moved.emit()

        return super().itemChange(change, value)

    def boundingRect(self):
        return QtCore.QRectF(
          0,0,
          NODE_WIDTH,
          NODE_PORT_Y + (len(self.inputPorts)+len(self.outputPorts))*NODE_PORT_SPACE - 10
        )

    def paint(self, painter, options, widget):

        br = self.boundingRect()
        br2 = QtCore.QRect(br.x(),br.y(),br.width(),NODE_HEADER_HEIGHT)
        path = QtGui.QPainterPath()
        path.addRect(br2)
        if self.isSelected():
            painter.fillPath(path,QtGui.QBrush(COLOR_RED_T))
        else:
            painter.fillPath(path,QtGui.QBrush(COLOR_BLUE_T))

        br2 = QtCore.QRect(br.x(),br.y()+NODE_HEADER_HEIGHT,br.width(),br.height()-NODE_HEADER_HEIGHT)
        path = QtGui.QPainterPath()
        path.addRect(br2)
        painter.fillPath(path,QtGui.QBrush(COLOR_BASE_T))

        # if self.isSelected():
        #     pen = QtGui.QPen(COLOR_BORDER, NODE_BORDER_WIDTH)
        #     painter.setPen(pen)
        #     path = QtGui.QPainterPath()
        #     path.addRect(br)
        #     painter.drawPath(path)


        # path.addRoundedRect(br,NODE_BORDER_RADIUS,NODE_BORDER_RADIUS)

        # path = QtGui.QPainterPath()
        # path.addRect(br)
        # painter.fillPath(path,QtGui.QBrush(COLOR_BASE_T))
        # painter.fillPath(path,COLOR_BASE_T)
        # painter.fillPath(path,COLOR_BASE_T)

        # pen = QtGui.QPen(COLOR_BORDER, NODE_BORDER_WIDTH)

        # painter.setPen(pen)
        # painter.drawPath(path)
