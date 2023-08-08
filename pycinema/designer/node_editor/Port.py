from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.designer.node_editor.NodeEditorStyle import *
from pycinema.designer.node_editor.InputText import InputText

class PortDisc(QtWidgets.QGraphicsEllipseItem):

    port_connection_line = QtWidgets.QGraphicsLineItem()

    def __init__(self,parent):
        super().__init__(parent)
        self.setPen(QtGui.QPen(COLOR_NORMAL, 0))
        self.setBrush(COLOR_NORMAL)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def mousePressEvent(self,event):
        pos = self.mapToScene(self.boundingRect().center())
        line = self.port_connection_line.line()
        line.setP1(pos)
        line.setP2(pos)
        self.port_connection_line.setLine(line)
        self.port_connection_line.show()
        return

    def mouseMoveEvent(self,event):
        line = self.port_connection_line.line()
        line.setP2( self.mapToScene(event.pos()) )
        self.port_connection_line.setLine(line)
        return

    def mouseReleaseEvent(self,event):
        self.port_connection_line.hide()

        s = self.parentItem().port
        t = None
        items = self.scene().items(event.scenePos());
        for i in items:
            if isinstance(i,Port):
                t = i.port
                break

        if not t or s.is_input == t.is_input or s.parent==t.parent:
            return

        if t.is_input:
            s,t = t,s

        s.set( t )

        return

PortDisc.port_connection_line.setPen(QtGui.QPen(COLOR_NORMAL, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
PortDisc.port_connection_line.setZValue(1000)
PortDisc.port_connection_line.hide()

class WidgetFrame(QtWidgets.QGraphicsRectItem):
    def __init__(self, width, height, parent=None):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.setRect(QtCore.QRect(0,0,self.width,self.height))

    def paint(self, painter, option, widget):
        br = self.boundingRect()
        path = QtGui.QPainterPath()
        path.addRoundedRect(br,3,3)
        painter.fillPath(path,QtGui.QBrush(COLOR_WIDGET))


class Port(QtWidgets.QGraphicsItem):

    def __init__(self,parent,port):
        super().__init__(parent)
        self.port = port

        parentBR = parent.boundingRect()

        self.widget_ = InputText(port, parentBR.width()-2*(PORT_SIZE))
        self.widget = QtWidgets.QGraphicsProxyWidget(self)
        self.widget.setWidget(self.widget_)

        self.widget.setZValue(Z_NODE_LAYER+2)
        portLabelBR = self.widget.boundingRect()
        if port.is_input:
          self.widget.setPos(
            PORT_SIZE,
            -portLabelBR.height()/2-1
          )
        else:
          self.widget.setPos(
            -PORT_SIZE-portLabelBR.width(),
            -portLabelBR.height()/2-1
          )

        self.widgetFrame = WidgetFrame(portLabelBR.width(), portLabelBR.height(), self)
        self.widgetFrame.setZValue(Z_NODE_LAYER+1)
        self.widgetFrame.setPos(self.widget.pos())

        self.disc = PortDisc(self)
        self.disc.setRect(-PORT_SIZE/2,-PORT_SIZE/2,PORT_SIZE,PORT_SIZE)

    def boundingRect(self):
        return self.widget.boundingRect().united(self.disc.boundingRect())

    def paint(self, painter, option, widget):
        return
