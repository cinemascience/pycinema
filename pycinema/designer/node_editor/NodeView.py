from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.designer.View import View
from pycinema.designer.FilterBrowser import FilterBrowser

import pycinema
import pycinema.filters

use_pgv = True
try:
    import pygraphviz as pgv
except:
    use_pgv = False
import igraph

from pycinema.designer.node_editor.NodeEditorStyle import *
from pycinema.designer.node_editor.Edge import Edge
from pycinema.designer.node_editor.Port import Port, PortDisc
from pycinema.designer.node_editor.Node import Node

class _NodeView(QtWidgets.QGraphicsView):

    mouse_pos0 = None
    mouse_pos1 = None

    def __init__(self):
        super().__init__()
        self._zoom = 1
        self._empty = True

        self.timer = None
        self.auto_layout = True
        self.auto_connect = True
        self.skip_layout_animation = False

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

        pycinema.Filter.on('filter_created', self.createNode)
        pycinema.Filter.on('filter_deleted', self.deleteNode)
        pycinema.Filter.on('connection_added', self.addEdge)
        pycinema.Filter.on('connection_removed', self.removeEdge)

    def addEdge(self, ports):
        self._scene.addItem(Edge(
          Port.port_map[ports[0]],
          Port.port_map[ports[1]]
        ))

        self.computeLayout()

    def removeEdge(self, ports):
        edge = Edge.edge_map[(ports[0],ports[1])]
        self._scene.removeItem(edge)
        del Edge.edge_map[(ports[0],ports[1])]


    def createNode(self,filter):
        node = Node(filter)

        br = self._scene.itemsBoundingRect()
        node.setPos(br.right()+10,0)

        self._scene.addItem(node)

        selectedItems = self._scene.selectedItems()
        if len(selectedItems)>0:
            self.autoConnect(selectedItems[0],node)

        self._scene.clearSelection()
        node.setSelected(True)

        self.computeLayout()

    def deleteNode(self,filter):
        node = Node.node_map[filter]
        self._scene.removeItem(node)
        self.computeLayout()

    def autoConnect(self,node_a,node_b,force=False):
        if not node_a or not node_b or (not force and not self.auto_connect):
            return

        for o_portName, o_port in node_a.filter.outputs.ports():
            for i_portName, i_port in node_b.filter.inputs.ports():
                if o_portName==i_portName:
                    bridge_ports = list(o_port.connections)
                    i_port.set(o_port)

                    for bridge_port in bridge_ports:
                        for o_portName_, o_port_ in node_b.filter.outputs.ports():
                            if bridge_port.name == o_portName_:
                                bridge_port.set(o_port_)

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
        super().keyPressEvent(event)
        if event.isAccepted():
            return

        if event.modifiers() == QtCore.Qt.ControlModifier:
            if event.key()==70:
                dialog = FilterBrowser()
                dialog.exec()
            elif event.key()==76:
                self.computeLayout(True)
        elif event.key()==32:
            self.fitInView()
        elif event.key()==QtCore.Qt.Key_Delete:
            for node in list(self._scene.selectedItems()):
                node.filter.delete()

    def computeLayout(self, force=False):
        if not force and not self.auto_layout:
            return

        filters = pycinema.Filter._filters

        if use_pgv:
            node_string = ''
            edge_string = ''
            for key in filters:
                filter = filters[key]

                i_port_string = ''
                o_port_string = ''

                node = Node.node_map[filter]
                br = node.boundingRect()

                width = br.width() / 72 # POINTS_PER_INCH
                height = (br.height()+40) / 72

                for name, _ in filter.inputs.ports():
                    i_port_string += '<i_' + name + '>|'
                for name, _ in filter.outputs.ports():
                    o_port_string += '<o_' + name + '>|'

                node_string += filter.id + '[shape=record, label="{{' + i_port_string[:-1] + '}|{' + o_port_string[:-1] + '}}",' + 'width=' + str(width) + ',' + 'height=' + str(height) + '];\n'

                for name, o_port in filter.outputs.ports():
                    for i_port in o_port.connections:
                        edge_string += filter.id + ':<o_' + name + '> -> ' + i_port.parent.id + ':<i_' +i_port.name+ '>;\n'

            edges = {}
            pycinema.Filter.computeDAG(edges)
            for s in edges:
                for t in edges[s]:
                    edge_string += s.id + "->" + t.id + '[style = invis, weight= 10];\n'

            dotString = 'digraph g {\nrankdir=LR;overlap = true;splines = true;graph[pad="0.5", ranksep="0.5", nodesep="0.5"];\n' + node_string + edge_string + '}'
            if pycinema.Filter._debug:
                print(dotString)

            G = pgv.AGraph(dotString)
            G.layout(prog='dot')

            for key in filters:
                filter = filters[key]
                node_ = G.get_node(str(filter.id))
                pos = [float(x) for x in node_.attr['pos'].split(',')]
                node = Node.node_map[filter]
                node.target = QtCore.QPointF(pos[0],-pos[1])
        else:
            g = igraph.Graph(directed=True)
            vertices = [f for f in filters]
            g.add_vertices( [f.id for f in vertices] )

            edges = {}
            pycinema.Filter.computeDAG(edges)
            L = pycinema.Filter.computeTopologicalOrdering(edges)

            edges_ = []
            edgesR = {}
            for n in edges:
              for m in edges[n]:
                if not m in edgesR:
                  edgesR[m] = set({})
                edgesR[m].add(n)
                edges_.append((n.id,m.id))
            g.add_edges(edges_)

            layout = g.layout_reingold_tilford(mode="out")
            scale = 250
            for i, f in enumerate(vertices):
                node = Node.node_map[f]
                coords = layout[i]
                Node.node_map[f].target = QtCore.QPointF(coords[1]*scale,-coords[0]*scale*0.7)

            for f in L:
                if not f in edgesR:
                    continue
                previous_filters = edgesR[f]
                max_x = -900000000
                for pf in previous_filters:
                    pn = Node.node_map[f]
                    max_x = max(max_x, Node.node_map[pf].target.x() + pn.boundingRect().width())

                Node.node_map[f].target.setX(max_x+50)

        if self.timer:
              self.timer.stop()
              self.timer.setParent(None)
              self.timer = None
        if self.skip_layout_animation:
            for key in filters:
              filter = filters[key]
              node = Node.node_map[filter]
              node.setPos(node.target)
        else:
            self.timer = QtCore.QTimeLine(300,self)
            self.timer.setFrameRange(0, 1)
            for key in filters:
                filter = filters[key]
                node = Node.node_map[filter]

                animation = QtWidgets.QGraphicsItemAnimation(node)
                animation.setItem(node)
                animation.setTimeLine(self.timer)
                animation.setPosAt(0, node.pos())
                animation.setPosAt(1, node.target)
            self.timer.start()

class NodeView(View):

    def __init__(self):
        super().__init__()
        self.setTitle(self.__class__.__name__)

        self.view = _NodeView()

        # remove close button
        self.button_c.setParent(None)

        self.content.layout().addWidget(self.view,1)
