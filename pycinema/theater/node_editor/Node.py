from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.node_editor.NodeEditorStyle import NodeEditorStyle as NES
from pycinema.theater.node_editor.Port import Port

class Node(QtWidgets.QGraphicsObject):

    s_moved = QtCore.Signal(name='s_moved')

    def __init__(self,filter,node_map,port_map):
        super().__init__()

        self.setZValue(NES.Z_NODE_LAYER)

        self.node_map = node_map
        self.port_map = port_map

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
        label_.setStyleSheet("background-color: transparent; color: "+NES.COLOR_NORMAL_);
        self.label.setWidget(label_)

        brL = self.label.boundingRect()

        nodeWidthToLabelWidthRatio = brN.width() / brL.width();
        if nodeWidthToLabelWidthRatio < 1.0:
            self.label.setScale(nodeWidthToLabelWidthRatio);
        else:
            nodeWidthToLabelWidthRatio = 1.0

        self.label.setPos(
          brN.center().x()-brL.width()*nodeWidthToLabelWidthRatio/2,
          NES.NODE_HEADER_HEIGHT/2 - brL.height()/2
        )

        # output ports
        portY = NES.NODE_PORT_Y

        for (ports,x) in [(self.outputPorts,brN.right()),(self.inputPorts,brN.left())]:
          for name, port in ports:
            qPort = Port(self,port)
            self.port_map[port] = qPort
            qPort.setPos(x,portY)
            portY += NES.NODE_PORT_SPACE

    def delete(self):
      for ports in [self.outputPorts,self.inputPorts]:
        for name, port in ports:
          del self.port_map[port]
      del self.node_map[self.filter]

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.s_moved.emit()

        return super().itemChange(change, value)

    def boundingRect(self):
        return QtCore.QRectF(
          0,0,
          NES.NODE_WIDTH,
          NES.NODE_PORT_Y + (len(self.inputPorts)+len(self.outputPorts))*NES.NODE_PORT_SPACE - 10
        )

    def paint(self, painter, options, widget):

        br = self.boundingRect()

        corner_radius = 5
        corner_radius2 = corner_radius*2

        path = QtGui.QPainterPath()
        path.moveTo(br.x(),br.y()+NES.NODE_HEADER_HEIGHT);
        path.lineTo(br.x()+br.width(),br.y()+NES.NODE_HEADER_HEIGHT);
        path.lineTo(br.x()+br.width(),br.y()+corner_radius);
        path.arcTo(br.x()+br.width()-corner_radius2,br.y(), corner_radius2, corner_radius2, 0.0, 90.0);
        path.lineTo(br.x()+corner_radius,br.y());
        path.arcTo(br.x(),br.y(), corner_radius2, corner_radius2, 90.0, 90.0);
        path.lineTo(br.x(),br.y()+corner_radius);

        if self.isSelected():
            painter.fillPath(path,QtGui.QBrush(NES.COLOR_RED_T))
        else:
            painter.fillPath(path,QtGui.QBrush(NES.COLOR_BLUE_T))

        path = QtGui.QPainterPath()

        path.moveTo(br.x(),br.y()+NES.NODE_HEADER_HEIGHT);
        path.lineTo(br.x(),br.y()+br.height()-corner_radius);
        path.arcTo(br.x(),br.y()+br.height()-corner_radius2,corner_radius2,corner_radius2,180,90);
        path.lineTo(br.x()+br.width()-corner_radius,br.y()+br.height());
        path.arcTo(br.x()+br.width()-corner_radius2,br.y()+br.height()-corner_radius2,corner_radius2,corner_radius2,270,90);
        path.lineTo(br.x()+br.width(),br.y()+NES.NODE_HEADER_HEIGHT);

        painter.fillPath(path,QtGui.QBrush(NES.COLOR_BASE_T))
