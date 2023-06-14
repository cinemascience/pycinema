from PySide6 import QtCore, QtWidgets, QtGui

from .View import *
from .FilterBrowser import *

from pycinema import Filter
import pycinema.filters

import pygraphviz as pgv

from .NodeEditorStyle import *

from .Edge import Edge
from .Port import Port, PortDisc
from .Node import Node

# COLOR_BASE = QtWidgets.QApplication.palette().dark().color()
# COLOR_BORDER = QtWidgets.QApplication.palette().mid().color()

class _NodeView(QtWidgets.QGraphicsView):

    last_added_node = None
    mouse_pos0 = None
    mouse_pos1 = None

    def __init__(self):
        super().__init__()
        self._zoom = 1
        self._empty = True

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

        self.node_connection_line = QtWidgets.QGraphicsLineItem()
        self.node_connection_line.setPen(QtGui.QPen(COLOR_NORMAL, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        self.node_connection_line.setZValue(1000)
        self._scene.addItem(self.node_connection_line)
        self.node_connection_line.hide()

        self._scene.addItem(PortDisc.port_connection_line)

        Filter.on('filter_created', self.createNode)
        Filter.on('connection_added', self.addEdge)
        Filter.on('connection_removed', self.removeEdge)

    def addEdge(self, ports):
        self._scene.addItem(Edge(
          Port.port_map[ports[0]],
          Port.port_map[ports[1]]
        ))

        self.computeLayout()

    def removeEdge(self, ports):
        # print("remove")
        # print(ports[0],Port.port_map[ports[0]])
        # print(ports[1],Port.port_map[ports[1]])

        edge = Edge.edge_map[(ports[0],ports[1])]
        self._scene.removeItem(edge)
        del Edge.edge_map[(ports[0],ports[1])]


    def createNode(self,filter):
        node = Node(filter)

        br = self._scene.itemsBoundingRect()
        node.setPos(br.right()+10,0)

        self._scene.addItem(node)

        self.autoConnect(_NodeView.last_added_node,node)

        _NodeView.last_added_node = node

        self.computeLayout()

    # def drawBackground(self,painter,rect):
    #     gridSize = 25

    #     left = rect.left() - rect.left() % gridSize
    #     top = rect.top() - rect.top() % gridSize

    #     lines = []

    #     for i in range(int(left),int(rect.right()),gridSize):
    #         lines.append(QtCore.QLineF(i, rect.top(), i, rect.bottom()))
    #     for i in range(int(top),int(rect.bottom()),gridSize):
    #         lines.append(QtCore.QLineF(rect.left(), i, rect.right(), i))

    #     painter.setBrush(QtGui.QBrush(COLOR_BACKGROUND))
    #     painter.drawRect(rect.adjusted(-10, -10, 10, 10))

    #     painter.setPen(QtGui.QPen(COLOR_GRID, 1))
    #     painter.drawLines(lines)

    def autoConnect(self,node_a,node_b):
        if not node_a or not node_b:
            return

        for oport in node_a.filter.outputs.ports():
            for iport in node_b.filter.inputs.ports():
                if oport==iport:
                    getattr(node_b.filter.inputs, iport).set(
                        getattr(node_a.filter.outputs, oport )
                    )
        return

    def fitInView(self):
        rect = QtCore.QRectF(self._scene.itemsBoundingRect())
        if rect.isNull():
            return

        self.resetTransform()
        self._zoom = 1.0
        self.centerOn(rect.center())

    def wheelEvent(self, event):
        ZOOM_INCREMENT_RATIO = 0.1

        angle = event.angleDelta().y()
        factor = 1.0
        if angle > 0:
            factor += ZOOM_INCREMENT_RATIO
        else:
            factor -= ZOOM_INCREMENT_RATIO

        self._zoom *= factor
        if self._zoom>3.0 or self._zoom<0.2:
            self._zoom /= factor
        else:
            self.scale(factor, factor)

    def focusInEvent(self,event):
        self.setStyleSheet("border:3px solid #15a3b4")

    def focusOutEvent(self,event):
        self.setStyleSheet("border:0 solid #15a3b4")

    def mousePressEvent(self,event):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            self.mouse_pos0 = self.mapToScene(event.pos())
            self.node_connection_line.show()
            line = self.node_connection_line.line()
            line.setP1(self.mouse_pos0)
            line.setP2(self.mouse_pos0)
            self.node_connection_line.setLine(line)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self.mouse_pos0:
            line = self.node_connection_line.line()
            line.setP2(self.mapToScene(event.pos()))
            self.node_connection_line.setLine(line)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self.mouse_pos0:

            self.node_connection_line.hide()
            self.mouse_pos0 = None

            line = self.node_connection_line.line()
            items0 = [x for x in self.scene().items(line.p1()) if isinstance(x,Node)]
            items1 = [x for x in self.scene().items(line.p2()) if isinstance(x,Node)]

            if len(items0)>0 and len(items1)>0 and items0[0]!=items1[0]:
                self.autoConnect(items0[0],items1[0])

        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self,event):

        if event.key()==32:
            if event.modifiers() == QtCore.Qt.ControlModifier:
                dialog = FilterBrowser()
                dialog.exec()
            elif event.modifiers() == QtCore.Qt.ShiftModifier:
                self.computeLayout()
            else:
                self.fitInView()

    def computeLayout(self):

        filters = pycinema.Filter._filters

        node_string = ''
        edge_string = ''
        for key in filters:
            filter = filters[key]

            i_port_string = ''
            o_port_string = ''

            node = Node.node_map[filter]
            br = node.boundingRect()

            width = br.width() / 72 # POINTS_PER_INCH
            height = br.height() / 72

            for name in filter.inputs.ports():
                i_port_string += '<i_' + name + '>|'
            for name in filter.outputs.ports():
                o_port_string += '<o_' + name + '>|'

            node_string += filter.id + '[shape=record, label="{{' + i_port_string[:-1] + '}|{' + o_port_string[:-1] + '}}",' + 'width=' + str(width) + ',' + 'height=' + str(height) + '];\n'

            for name in filter.outputs.ports():
                port = getattr(filter.outputs, name)
                for iport in port.connections:
                    edge_string += filter.id + ':<o_' + name + '> -> ' + iport.parent.id + ':<i_' +iport.name+ '>;\n'

        edges = {}
        Filter.computeDAG(edges)
        for s in edges:
            for t in edges[s]:
                edge_string += s.id + "->" + t.id + '[style = invis, weight= 10];\n'

        dotString = 'digraph g {\nrankdir=LR;overlap = true;splines = true;graph[pad="0.5", ranksep="0.5", nodesep="0.5"];\n' + node_string + edge_string + '\n}'
        if Filter._debug:
            print(dotString)

        G = pgv.AGraph(dotString)
        G.layout(prog='dot')
        for key in filters:
            filter = filters[key]
            node_ = G.get_node(str(filter.id))
            pos = [float(x) for x in node_.attr['pos'].split(',')]
            node = Node.node_map[filter]
            # node.setPos(pos[0],pos[1])

            timer = QtCore.QTimeLine(300,node)
            timer.setFrameRange(0, 1)
            animation = QtWidgets.QGraphicsItemAnimation(node)
            animation.setItem(node)
            animation.setTimeLine(timer)
            animation.setPosAt(0, node.pos())
            animation.setPosAt(1, QtCore.QPointF(pos[0],pos[1]))
            timer.start()


        # dot = graphviz.Source(dotString)
        # dot.

        # dot = graphviz.Digraph(comment='The Round Table')

        # dot.node('A', 'King Arthur')  # doctest: +NO_EXE
        # dot.node('B', 'Sir Bedevere the Wise')

        # dot.edges(['AB', 'AL'])

class NodeView(View):

    def __init__(self):
        super().__init__()
        self.setTitle(self.__class__.__name__)

        self.view = _NodeView()

        # remove close button
        self.button_c.setParent(None)

        self.content.layout().addWidget(self.view,1)

        # # Filter._debug = True
        # x = pycinema.filters.CinemaDatabaseReader()
        # # x.inputs.path.set('/home/jones/external/projects/cinema-lib/pycinema-data/Warp.cdb')
        # x.inputs.path.set('/home/jones/external/projects/cinema-lib/pycinema/data/ScalarImages.cdb')

        # y = pycinema.filters.DatabaseQuery()
        # y.inputs.table.set(x.outputs.table)
        # y.inputs.sql.set('SELECT * FROM input LIMIT 3')


        # z = pycinema.filters.ImageReader()
        # z.inputs.table.set(y.outputs.table)

        # w = pycinema.filters.ColorMapping()
        # w.inputs.channel.set('isovalue')
        # w.inputs.map.set('RdYlBu')
        # w.inputs.range.set((-9,9))
        # w.inputs.images.set(z.outputs.images)

        # z = pycinema.ImageReader()
        # pycinema.Filter.trigger('filter_created',z)


