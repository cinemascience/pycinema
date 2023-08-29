from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.node_editor.NodeEditorStyle import NodeEditorStyle as NES
from pycinema.theater.node_editor.InputText import InputText

class PortDisc(QtWidgets.QGraphicsEllipseItem):

    port_connection_line = QtWidgets.QGraphicsLineItem()

    def __init__(self,parent):
        super().__init__(parent)

        if PortDisc.port_connection_line.zValue()<1000:
          PortDisc.port_connection_line.setPen(QtGui.QPen(NES.COLOR_NORMAL, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
          PortDisc.port_connection_line.setZValue(1000)
          PortDisc.port_connection_line.hide()

        self.setPen(QtGui.QPen(NES.COLOR_NORMAL, 0))
        self.setBrush(NES.COLOR_NORMAL)
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
        painter.fillPath(path,QtGui.QBrush(NES.COLOR_WIDGET_T))


class Port(QtWidgets.QGraphicsItem):

    def __init__(self,parent,port):
        super().__init__(parent)
        self.port = port

        parentBR = parent.boundingRect()

        self.widget_ = InputText(port, parentBR.width()-2*(NES.PORT_SIZE))
        self.widget = QtWidgets.QGraphicsProxyWidget(self)
        self.widget.setWidget(self.widget_)

        self.widget.setZValue(NES.Z_NODE_LAYER+2)
        portLabelBR = self.widget.boundingRect()
        if port.is_input:
          self.widget.setPos(
            NES.PORT_SIZE,
            -portLabelBR.height()/2-1
          )
        else:
          self.widget.setPos(
            -NES.PORT_SIZE-portLabelBR.width(),
            -portLabelBR.height()/2-1
          )

        self.widgetFrame = WidgetFrame(portLabelBR.width(), portLabelBR.height(), self)
        self.widgetFrame.setZValue(NES.Z_NODE_LAYER+1)
        self.widgetFrame.setPos(self.widget.pos())

        self.disc = PortDisc(self)
        self.disc.setRect(-NES.PORT_SIZE/2,-NES.PORT_SIZE/2,NES.PORT_SIZE,NES.PORT_SIZE)

    def boundingRect(self):
        return self.widget.boundingRect().united(self.disc.boundingRect())

    def paint(self, painter, option, widget):
        return
