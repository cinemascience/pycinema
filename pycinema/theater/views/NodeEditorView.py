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

from pycinema.theater.node_editor.NodeEditorStyle import NodeEditorStyle as NES
from pycinema.theater.node_editor.Edge import Edge
from pycinema.theater.node_editor.Port import Port, PortDisc
from pycinema.theater.node_editor.Node import Node

class QtNodeEditorView(QtWidgets.QGraphicsView):

    mouse_pos0 = None
    mouse_pos1 = None

    scene = None
    node_map = {}
    edge_map = {}
    port_map = {}

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

        if not QtNodeEditorView.scene:
          QtNodeEditorView.init_gobal()
        self.setScene(QtNodeEditorView.scene)

    def init_gobal():
        node_connection_line = QtWidgets.QGraphicsLineItem()
        node_connection_line.setPen(QtGui.QPen(NES.COLOR_NORMAL, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        node_connection_line.setZValue(1000)
        node_connection_line.hide()

        QtNodeEditorView.scene = QtWidgets.QGraphicsScene()
        QtNodeEditorView.scene.addItem(node_connection_line)
        QtNodeEditorView.scene.node_connection_line = node_connection_line

        pycinema.Filter.on('filter_created', QtNodeEditorView.addNode)
        pycinema.Filter.on('filter_deleted', QtNodeEditorView.removeNode)
        pycinema.Filter.on('connection_added', QtNodeEditorView.addEdge)
        pycinema.Filter.on('connection_removed', QtNodeEditorView.removeEdge)

    def addEdge(ports):
        p0 = QtNodeEditorView.port_map[ports[0]]
        p1 = QtNodeEditorView.port_map[ports[1]]
        edge = Edge(p0,p1)
        QtNodeEditorView.edge_map[(ports[0],ports[1])] = edge
        QtNodeEditorView.scene.addItem(edge)
        QtNodeEditorView.computeLayout()

    def removeEdge(ports):
        edge = QtNodeEditorView.edge_map[(ports[0],ports[1])]
        QtNodeEditorView.scene.removeItem(edge)
        del QtNodeEditorView.edge_map[(ports[0],ports[1])]
        QtNodeEditorView.computeLayout()

    def addNode(filter):
        node = Node(filter,QtNodeEditorView.node_map,QtNodeEditorView.port_map)

        br = QtNodeEditorView.scene.itemsBoundingRect()
        node.setPos(br.right()+10,0)
        QtNodeEditorView.scene.addItem(node)

        selectedItems = QtNodeEditorView.scene.selectedItems()
        if len(selectedItems)>0:
            QtNodeEditorView.autoConnect(selectedItems[0],node)

        QtNodeEditorView.scene.clearSelection()
        node.setSelected(True)

        QtNodeEditorView.computeLayout()

    def removeNode(filter):
        node = QtNodeEditorView.node_map[filter]
        QtNodeEditorView.scene.removeItem(node)
        node.delete()
        QtNodeEditorView.computeLayout()

    def autoConnect(node_a,node_b,force=False):
        if not node_a or not node_b or (not force and not QtNodeEditorView.auto_connect):
            return

        node_a_ports = node_a.filter.outputs.ports()
        node_b_ports = node_b.filter.inputs.ports()

        if len(node_b_ports)==1:
          for o_portName, o_port in node_a_ports:
              for i_portName, i_port in node_b_ports:
                  bridge_ports = list(o_port.connections)
                  i_port.set(o_port)
                  for bridge_port in bridge_ports:
                      if bridge_port.parent==node_b.filter: continue
                      for o_portName_, o_port_ in node_b.filter.outputs.ports():
                          if bridge_port.name == o_portName_:
                              bridge_port.set(o_port_)
                  return

        for o_portName, o_port in node_a_ports:
            for i_portName, i_port in node_b_ports:
                if o_portName==i_portName:
                    bridge_ports = list(o_port.connections)
                    i_port.set(o_port)
                    for bridge_port in bridge_ports:
                        if bridge_port.parent==node_b.filter: continue
                        for o_portName_, o_port_ in node_b.filter.outputs.ports():
                            if bridge_port.name == o_portName_:
                                bridge_port.set(o_port_)

    def fitInView(self):
        rect = QtCore.QRectF(QtNodeEditorView.scene.itemsBoundingRect())
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
            QtNodeEditorView.mouse_pos0 = self.mapToScene(event.pos())
            ncl = QtNodeEditorView.scene.node_connection_line
            ncl.show()
            line = ncl.line()
            line.setP1(QtNodeEditorView.mouse_pos0)
            line.setP2(QtNodeEditorView.mouse_pos0)
            ncl.setLine(line)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if QtNodeEditorView.mouse_pos0:
            ncl = QtNodeEditorView.scene.node_connection_line
            line = ncl.line()
            line.setP2(self.mapToScene(event.pos()))
            ncl.setLine(line)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if QtNodeEditorView.mouse_pos0:

            ncl = QtNodeEditorView.scene.node_connection_line
            ncl.hide()
            QtNodeEditorView.mouse_pos0 = None

            line = ncl.line()
            items0 = [x for x in QtNodeEditorView.scene.items(line.p1()) if isinstance(x,Node)]
            items1 = [x for x in QtNodeEditorView.scene.items(line.p2()) if isinstance(x,Node)]

            if len(items0)>0 and len(items1)>0 and items0[0]!=items1[0]:
                QtNodeEditorView.autoConnect(items0[0],items1[0])

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
                QtNodeEditorView.computeLayout(True)
        elif event.key()==32:
            self.fitInView()
        elif event.key()==QtCore.Qt.Key_Delete:
            for node in list(QtNodeEditorView.scene.selectedItems()):
                iFilters = node.filter.getInputFilters()
                oFilters = node.filter.getOutputFilters()
                node.filter.delete()
                for iFilter in iFilters:
                  for oFilter in oFilters:
                    QtNodeEditorView.autoConnect(
                      QtNodeEditorView.node_map[iFilter],
                      QtNodeEditorView.node_map[oFilter]
                    )

    def computeLayout(force=False):
        if not force and not QtNodeEditorView.auto_layout:
            return

        filters = pycinema.Filter._filters

        if use_pgv:
            node_string = ''
            edge_string = ''
            for key in filters:
                filter = filters[key]

                i_port_string = ''
                o_port_string = ''

                node = QtNodeEditorView.node_map[filter]
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
                node = QtNodeEditorView.node_map[filter]
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
                node = QtNodeEditorView.node_map[f]
                coords = layout[i]
                QtNodeEditorView.node_map[f].target = QtCore.QPointF(coords[1]*scale,coords[0]*scale*0.7)

            for f in L:
                if not f in edgesR:
                    continue
                previous_filters = edgesR[f]
                max_x = -900000000
                for pf in previous_filters:
                    pn = QtNodeEditorView.node_map[f]
                    max_x = max(max_x, QtNodeEditorView.node_map[pf].target.x() + pn.boundingRect().width())

                QtNodeEditorView.node_map[f].target.setX(max_x+50)

        if QtNodeEditorView.timer:
              QtNodeEditorView.timer.stop()
              QtNodeEditorView.timer.setParent(None)
              QtNodeEditorView.timer = None
        if QtNodeEditorView.skip_layout_animation:
            for key in filters:
              filter = filters[key]
              node = QtNodeEditorView.node_map[filter]
              node.setPos(node.target)
        else:
            QtNodeEditorView.timer = QtCore.QTimeLine(300)
            QtNodeEditorView.timer.setFrameRange(0, 1)
            for key in filters:
                filter = filters[key]
                node = QtNodeEditorView.node_map[filter]

                animation = QtWidgets.QGraphicsItemAnimation(node)
                animation.setItem(node)
                animation.setTimeLine(QtNodeEditorView.timer)
                animation.setPosAt(0, node.pos())
                animation.setPosAt(1, node.target)
            QtNodeEditorView.timer.start()

class NodeEditorView(View):

    def __init__(self):
        super().__init__()
        self.setTitle(self.__class__.__name__)

        self.view = QtNodeEditorView()
        self.content.layout().addWidget(self.view,1)

        QtCore.QTimer.singleShot(0, self.view.fitInView)

