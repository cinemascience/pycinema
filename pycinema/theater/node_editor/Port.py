from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.node_editor.NodeEditorStyle import NodeEditorStyle as NES
from pycinema.theater.node_editor.InputText import InputText

class PortDisc(QtWidgets.QGraphicsEllipseItem):

    def __init__(self,parent):
        super().__init__(parent)

        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setZValue(1000)
        self.setAcceptHoverEvents(True)
        self.setPen(QtCore.Qt.NoPen)

        self.representation = QtWidgets.QGraphicsEllipseItem(self)
        # self.setPen(QtGui.QPen(NES.COLOR_NORMAL, 0))
        # self.setBrush(NES.COLOR_NORMAL)
        self.representation.setPen(QtCore.Qt.NoPen)
        self.representation.setBrush(NES.COLOR_NORMAL)

    def setRect(self,rect):
        super().setRect(rect.adjusted(-10,-10,10,10))
        self.representation.setRect(rect)
        # return super().setRect(rect)

    def hoverEnterEvent(self,event):
        self.representation.setBrush(NES.COLOR_BLUE)
    def hoverLeaveEvent(self,event):
        self.representation.setBrush(NES.COLOR_NORMAL)

    def mousePressEvent(self,event):
        pos = self.mapToScene(self.boundingRect().center())
        ncl = self.scene().node_connection_line
        line = ncl.line()
        line.setP1(pos)
        line.setP2(pos)
        ncl.setLine(line)
        ncl.show()

    def mouseMoveEvent(self,event):
        ncl = self.scene().node_connection_line
        line = ncl.line()
        line.setP2( self.mapToScene(event.pos()) )
        ncl.setLine(line)

    def mouseReleaseEvent(self,event):
        ncl = self.scene().node_connection_line
        ncl.hide()

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

    def mouseDoubleClickEvent(self,event):
        self.parentItem().port.set(self.parentItem().port.default)

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
        self.disc.setRect(
          QtCore.QRect(-NES.PORT_SIZE/2,-NES.PORT_SIZE/2,NES.PORT_SIZE,NES.PORT_SIZE)
        )

    def boundingRect(self):
        return self.widget.boundingRect().united(self.disc.boundingRect())

    def paint(self, painter, option, widget):
        return
