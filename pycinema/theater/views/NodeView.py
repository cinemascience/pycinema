from PySide6 import QtCore, QtWidgets, QtGui

import pycinema
import pycinema.filters

from pycinema.theater.View import View
from pycinema.theater.FilterBrowser import FilterBrowser

use_pgv = True
try:
    import pygraphviz as pgv
except:
    use_pgv = False
import igraph

from pycinema.theater.node_editor.NodeEditorStyle import *
from pycinema.theater.node_editor.Edge import Edge
from pycinema.theater.node_editor.Port import Port, PortDisc
from pycinema.theater.node_editor.Node import Node

import weakref

class QtNodeView(QtWidgets.QGraphicsView):

    instances = weakref.WeakKeyDictionary()

    mouse_pos0 = None
    mouse_pos1 = None

    scene = None
    node_map = {}
    edge_map = {}
    port_map = {}
    node_connection_line = None

    auto_layout = True
    auto_connect = True
    skip_layout_animation = False

    timer = None

    def __init__(self):
        super().__init__()

        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        l = 10000
        self.setSceneRect(-l,-l,2*l,2*l)

        if not QtNodeView.scene:
          QtNodeView.init_gobal()
        self.setScene(QtNodeView.scene)

        QtNodeView.instances[self] = None

    def init_gobal():
        QtNodeView.node_connection_line = QtWidgets.QGraphicsLineItem()
        QtNodeView.node_connection_line.setPen(QtGui.QPen(COLOR_NORMAL, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        QtNodeView.node_connection_line.setZValue(1000)
        QtNodeView.node_connection_line.hide()

        QtNodeView.scene = QtWidgets.QGraphicsScene()
        QtNodeView.scene.addItem(PortDisc.port_connection_line)
        QtNodeView.scene.addItem(QtNodeView.node_connection_line)

        pycinema.Filter.on('filter_created', QtNodeView.addNode)
        pycinema.Filter.on('filter_deleted', QtNodeView.removeNode)
        pycinema.Filter.on('connection_added', QtNodeView.addEdge)
        pycinema.Filter.on('connection_removed', QtNodeView.removeEdge)

    def addEdge(ports):
        p0 = QtNodeView.port_map[ports[0]]
        p1 = QtNodeView.port_map[ports[1]]
        edge = Edge(p0,p1)
        QtNodeView.edge_map[(ports[0],ports[1])] = edge
        QtNodeView.scene.addItem(edge)
        QtNodeView.computeLayout()

    def removeEdge(ports):
        edge = QtNodeView.edge_map[(ports[0],ports[1])]
        QtNodeView.scene.removeItem(edge)
        del QtNodeView.edge_map[(ports[0],ports[1])]
        QtNodeView.computeLayout()

    def addNode(filter):
        node = Node(filter,QtNodeView.node_map,QtNodeView.port_map)

        br = QtNodeView.scene.itemsBoundingRect()
        node.setPos(br.right()+10,0)
        QtNodeView.scene.addItem(node)

        selectedItems = QtNodeView.scene.selectedItems()
        if len(selectedItems)>0:
            QtNodeView.autoConnect(selectedItems[0],node)

        QtNodeView.scene.clearSelection()
        node.setSelected(True)

        QtNodeView.computeLayout()

    def removeNode(filter):
        node = QtNodeView.node_map[filter]
        QtNodeView.scene.removeItem(node)
        node.delete()
        QtNodeView.computeLayout()

    def autoConnect(node_a,node_b,force=False):
        if not node_a or not node_b or (not force and not QtNodeView.auto_connect):
            return

        for o_portName, o_port in node_a.filter.outputs.ports():
            for i_portName, i_port in node_b.filter.inputs.ports():
                if o_portName==i_portName:
                    bridge_ports = list(o_port.connections)
                    i_port.set(o_port)
                    for bridge_port in bridge_ports:
                        if bridge_port.parent==node_b.filter: continue
                        for o_portName_, o_port_ in node_b.filter.outputs.ports():
                            if bridge_port.name == o_portName_:
                                bridge_port.set(o_port_)

    def fitInView(self):
        rect = QtCore.QRectF(QtNodeView.scene.itemsBoundingRect())
        if rect.isNull():
            return

        self.resetTransform()
        super().fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        ZOOM_INCREMENT_RATIO = 0.1

        angle = event.angleDelta().y()
        factor = 1.0
        if angle > 0:
            factor += ZOOM_INCREMENT_RATIO
        else:
            factor -= ZOOM_INCREMENT_RATIO

        self.scale(factor, factor)

    def focusInEvent(self,event):
        self.setStyleSheet("border:3px solid #15a3b4")

    def focusOutEvent(self,event):
        self.setStyleSheet("border:0 solid #15a3b4")

    def mousePressEvent(self,event):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            QtNodeView.mouse_pos0 = self.mapToScene(event.pos())
            QtNodeView.node_connection_line.show()
            line = QtNodeView.node_connection_line.line()
            line.setP1(QtNodeView.mouse_pos0)
            line.setP2(QtNodeView.mouse_pos0)
            QtNodeView.node_connection_line.setLine(line)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if QtNodeView.mouse_pos0:
            line = QtNodeView.node_connection_line.line()
            line.setP2(self.mapToScene(event.pos()))
            QtNodeView.node_connection_line.setLine(line)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if QtNodeView.mouse_pos0:

            QtNodeView.node_connection_line.hide()
            QtNodeView.mouse_pos0 = None

            line = QtNodeView.node_connection_line.line()
            items0 = [x for x in QtNodeView.scene.items(line.p1()) if isinstance(x,Node)]
            items1 = [x for x in QtNodeView.scene.items(line.p2()) if isinstance(x,Node)]

            if len(items0)>0 and len(items1)>0 and items0[0]!=items1[0]:
                QtNodeView.autoConnect(items0[0],items1[0])

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
                QtNodeView.computeLayout(True)
        elif event.key()==32:
            self.fitInView()
        elif event.key()==QtCore.Qt.Key_Delete:
            for node in list(QtNodeView.scene.selectedItems()):
                node.filter.delete()

    def computeLayout(force=False):
        if not force and not QtNodeView.auto_layout:
            return

        filters = pycinema.Filter._filters

        if use_pgv:
            node_string = ''
            edge_string = ''
            for key in filters:
                filter = filters[key]

                i_port_string = ''
                o_port_string = ''

                node = QtNodeView.node_map[filter]
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
                node = QtNodeView.node_map[filter]
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
                node = QtNodeView.node_map[f]
                coords = layout[i]
                QtNodeView.node_map[f].target = QtCore.QPointF(coords[1]*scale,-coords[0]*scale*0.7)

            for f in L:
                if not f in edgesR:
                    continue
                previous_filters = edgesR[f]
                max_x = -900000000
                for pf in previous_filters:
                    pn = QtNodeView.node_map[f]
                    max_x = max(max_x, QtNodeView.node_map[pf].target.x() + pn.boundingRect().width())

                QtNodeView.node_map[f].target.setX(max_x+50)

        if QtNodeView.timer:
              QtNodeView.timer.stop()
              QtNodeView.timer.setParent(None)
              QtNodeView.timer = None
        if QtNodeView.skip_layout_animation:
            for key in filters:
              filter = filters[key]
              node = QtNodeView.node_map[filter]
              node.setPos(node.target)
        else:
            QtNodeView.timer = QtCore.QTimeLine(300)
            QtNodeView.timer.setFrameRange(0, 1)
            for key in filters:
                filter = filters[key]
                node = QtNodeView.node_map[filter]

                animation = QtWidgets.QGraphicsItemAnimation(node)
                animation.setItem(node)
                animation.setTimeLine(QtNodeView.timer)
                animation.setPosAt(0, node.pos())
                animation.setPosAt(1, node.target)
            QtNodeView.timer.start()

class NodeView(View):

    def __init__(self):
        super().__init__()
        self.setTitle(self.__class__.__name__)

        self.view = QtNodeView()
        self.content.layout().addWidget(self.view,1)
