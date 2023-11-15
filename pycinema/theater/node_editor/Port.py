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
        self.representation.setPen(QtCore.Qt.NoPen)
        self.representation.setBrush(NES.COLOR_NORMAL)

    def setRect(self,rect):
        self.representation.setRect(rect)
        return super().setRect(rect.adjusted(-10,-10,10,10))

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

        s = self.parentItem().port
        t = None
        items = self.scene().items(event.scenePos());
        for i in items:
            if isinstance(i,Port):
                t = i
                break
        if t:
          if t.port.name=='images':
            print(t.port.parent)
            print(t.port.parent.inputs.images,t.port.parent.inputs.images.is_input)
        if not t or s.is_input == t.port.is_input or s.parent==t.port.parent:
          line.setP2( self.mapToScene(event.pos()) )
          ncl.setLine(line)
        else:
          line.setP2( t.disc.mapToScene(t.disc.boundingRect().center()) )
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
        if event.modifiers() == QtCore.Qt.ShiftModifier:
            if s.valueIsPortList():
                ports = [p for p in s._value]
                ports.append(t)
                s.set( ports )
            elif s.valueIsPort():
                s.set( [s._value,t] )
            else:
                s.set( t )
        else:
            s.set( t )

    def mouseDoubleClickEvent(self,event):
        self.parentItem().port.set(self.parentItem().port.default)

class InputTextGraphicsItem(QtWidgets.QGraphicsItem):
    def __init__(self, port, width, parent=None):
        super().__init__(parent)

        self.widget_ = InputText(port, width)
        self.widget = QtWidgets.QGraphicsProxyWidget(self)
        self.widget.setWidget(self.widget_)
        self.widget.setZValue(NES.Z_NODE_LAYER+2)

    def boundingRect(self):
        return self.widget.boundingRect()

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
        widget_width = parentBR.width()-2*(NES.PORT_SIZE)

        self.widget_ = InputText(port, widget_width)
        self.widget = QtWidgets.QGraphicsProxyWidget(self)
        self.widget.setWidget(self.widget_)
        self.widget.setZValue(NES.Z_NODE_LAYER+2)

        widget_br = self.widget.boundingRect()
        self.widget.setZValue(NES.Z_NODE_LAYER+1)
        if port.is_input:
          self.widget.setPos(
            NES.PORT_SIZE,
            -widget_br.height()/2-1
          )
        else:
          self.widget.setPos(
            -NES.PORT_SIZE-widget_br.width(),
            -widget_br.height()/2-1
          )

        self.disc = PortDisc(self)
        self.disc.setRect(
          QtCore.QRect(-NES.PORT_SIZE/2,-NES.PORT_SIZE/2,NES.PORT_SIZE,NES.PORT_SIZE)
        )

    def boundingRect(self):
        br = self.widget.boundingRect()
        br.moveTo(self.widget.pos())
        return br.united(self.disc.boundingRect())

    def paint(self, painter, option, widget):
        br = self.widget.boundingRect()
        br.moveTo(self.widget.pos())
        path = QtGui.QPainterPath()
        path.addRoundedRect(br,3,3)
        painter.fillPath(path,QtGui.QBrush(NES.COLOR_WIDGET_T))
        return
