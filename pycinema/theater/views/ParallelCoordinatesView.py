from PySide6 import QtCore, QtWidgets, QtGui

import sqlite3
import re

from pycinema.theater.views.FilterView import FilterView
from pycinema import Filter, getTableExtent, isNumber

def executeSQL(db,sql):
  try:
    c = db.cursor()
    c.execute(sql)
  except sqlite3.Error as e:
    print(e)

def createTable(db, table):
  sql = 'CREATE TABLE input(id INTEGER PRIMARY KEY AUTOINCREMENT';
  header = table[0]
  firstRow = table[1]
  for i in range(0,len(header)):
    if header[i].lower()=='id':
      continue
    sql = sql + ', ' + header[i] + ' TEXT';
  sql =  sql + ')';
  executeSQL(db,sql)

def insertData(db, table):
  sql = 'INSERT INTO input(';
  for x in table[0]:
    sql = sql + x + ', ';
  sql = sql[0:-2] + ') VALUES\n';

  for i in range(1, len(table)):
    row = '('
    for v in table[i]:
      row += '"' + str(v) + '",'
    sql += row[0:-1] + '),\n'
  sql = sql[0:-2];
  executeSQL(db,sql)

def queryData(db, sqlQuery):
  c = db.cursor()
  try:
    c.execute(sqlQuery)
  except sqlite3.Error as er:
    print('[SQL ERROR] %s' % (' '.join(er.args)))
    return [[]]
  res = c.fetchall()
  columns = []
  for d in c.description:
    columns.append(d[0])
  res.insert(0,columns)
  return res

def createLabel(parent,text):
  _label = QtWidgets.QLabel(str(text))
  font = QtGui.QFont()
  font.setPointSize(10)
  _label.setFont(font)

  palette = QtCore.QCoreApplication.instance().palette()
  color = 'color: '+palette.text().color().name()+';background:transparent;';
  _label.setStyleSheet(color);
  label = QtWidgets.QGraphicsProxyWidget(parent)
  label._color = color;
  label.setWidget(_label)
  return label

def computeValues(header,table):
  values = []
  for (p,idx) in header:
    v_dic = set()

    for j in range(1,len(table)):
      v_dic.add(table[j][idx])

    v_list = list(v_dic)
    isListOfNumbers = isNumber(table[1][idx])
    if isListOfNumbers:
      v_list = [(float(x),x) for x in v_list]
    v_list.sort()
    if isListOfNumbers:
      v_list = [x[1] for x in v_list]

    values.append(v_list)

  return values

class Lines(QtWidgets.QGraphicsItem):
  def __init__(self):
    super().__init__()
    self.br = QtCore.QRectF(0,0,0,0)

    palette = QtCore.QCoreApplication.instance().palette()

    # determine theme
    dark_theme = base_color = QtCore.QCoreApplication.instance().palette().base().color().red()<100
    if dark_theme:
      self.pen_line_normal = QtGui.QPen(QtGui.QColor("#555"))
      self.pen_line_highlight = QtGui.QPen(QtGui.QColor("#fff"))
    else:
      self.pen_line_normal = QtGui.QPen(QtGui.QColor("#eee"))
      self.pen_line_highlight = QtGui.QPen(QtGui.QColor("#333"))

    self.pen_line_highlight.setWidth(2)

    self.lines_normal = []
    self.lines_highlight = []

    self.path_normal = QtGui.QPainterPath()
    self.path_highlight = QtGui.QPainterPath()

  def setBoundingRect(self,br):
    self.prepareGeometryChange()
    self.br = br

    if len(self.lines_normal)<1: return

    n2 = len(self.lines_normal[0])
    n = n2-1

    x0 = self.br.x()
    y0 = self.br.y()
    w = self.br.width()
    h = self.br.height()

    interpolate = lambda l,q,q0: q0+l*q

    self.path_normal = QtGui.QPainterPath()
    self.path_highlight = QtGui.QPainterPath()

    for (lines,path) in [(self.lines_normal,self.path_normal),(self.lines_highlight,self.path_highlight)]:
      for line in lines:
        path.moveTo( x0, interpolate(line[0][1],h,y0) )
        for i in range(1,n2):
          path.lineTo( interpolate(line[i][0],w,x0), interpolate(line[i][1],h,y0) )

  def boundingRect(self):
    return self.br

  def paint(self, painter, option, widget):
    painter.setPen(self.pen_line_normal)
    painter.drawPath(self.path_normal)
    painter.setPen(self.pen_line_highlight)
    painter.drawPath(self.path_highlight)

class Axis(QtWidgets.QGraphicsItem):

  def __init__(self, parameter, values, emitter):
    super().__init__()

    self.br = QtCore.QRectF(0,0,0,0)

    self.emitter = emitter

    self.parameter = parameter
    self.values = values
    self.n_values = len(values)
    self.selection_idx0 = 0
    self.selection_idx1 = 0
    self.compose = False
    self.mouse_state = -1

    # label
    self.label = createLabel(self,parameter)
    self.label.setPos(-self.label.boundingRect().width()/2,0)
    self.label.widget().setCursor(QtCore.Qt.PointingHandCursor)

    # pens
    palette = QtCore.QCoreApplication.instance().palette()
    self.pen_grid = QtGui.QPen(palette.mid().color())

    # main line
    self.line = QtWidgets.QGraphicsLineItem(
      0, -100, 0, 100,
      self
    )
    self.line.setPen(self.pen_grid)

    # ticks
    self.ticks = []
    if len(self.values)==1:
      self.addTick(0.5,self.values[0])
    elif type(self.values[0])==str or len(self.values)<6:
      i = 0
      d = 1/(len(self.values)-1)
      for t in self.values:
        self.addTick(i,t)
        i += d
    else:
      return print('ERROR: unsupported axis type')

    # highlight bar
    self.bar = QtWidgets.QGraphicsRectItem(0,0,0,0,self)

    dark_theme = base_color = QtCore.QCoreApplication.instance().palette().base().color().red()<100
    if dark_theme:
      bar_c = QtGui.QColor('#fff')
    else:
      bar_c = QtGui.QColor('#333')

    self.bar.setBrush(bar_c)
    bar_pen = QtGui.QPen(bar_c)
    bar_pen.setWidth(3)
    self.bar.setPen(bar_pen)
    self.bar.setCursor(QtCore.Qt.PointingHandCursor)

  def snapToValueIdx(self,y):
    l = (y - self.y0) / (self.y1 - self.y0)
    l = max(min(l,1),0)
    return round(l*(self.n_values-1))

  def mousePressEvent(self,event):
    y = event.pos().y()
    if y<self.y0:
      self.mouse_state = 0
    else:
      self.mouse_state = 1
      idx = self.snapToValueIdx(y)
      self.selection_idx0 = idx
      self.selection_idx1 = idx
      self.emitter.s_selection_changed_intermediate.emit()

  def mouseMoveEvent(self,event):
    if self.mouse_state != 1: return
    idx = self.snapToValueIdx(event.pos().y())
    if idx!=self.selection_idx1:
      self.selection_idx1 = idx
      self.emitter.s_selection_changed_intermediate.emit()

  def mouseReleaseEvent(self,event):
    if self.mouse_state == 0:
      self.compose = not self.compose
    self.mouse_state = -1
    self.emitter.s_selection_changed.emit()

  def addTick(self, l, text):
    tick_line = QtWidgets.QGraphicsLineItem(0,0,0,0,self)
    tick_line.l = l
    tick_line.setPen(self.pen_grid)

    tick_label = createLabel(self,text)
    self.ticks.append((tick_line,tick_label))

  def update(self, x, y, height, axis_width):
    self.prepareGeometryChange()

    self.br = QtCore.QRectF(-axis_width/2,y,axis_width,height)

    self.setPos(x,0)
    self.label.setPos(self.label.pos().x(),y+5)

    self.y0 = y+30
    self.y1 = y+height-10

    self.line.setLine(0,self.y0,0,self.y1)

    for (tick_line,tick_label) in self.ticks:
      y = (1-tick_line.l)*self.y0 + tick_line.l*self.y1
      tick_line.setLine(-5,y,5,y)
      lb = tick_label.boundingRect()
      tick_label.setPos(-10-lb.width(),y-lb.height()/2)

    if self.selection_idx1 < self.selection_idx0:
      idx0, idx1 = self.selection_idx1, self.selection_idx0
    else:
      idx0, idx1 = self.selection_idx0, self.selection_idx1

    idx0 = max(0,idx0-0.1)
    idx1 = min(self.n_values-1,idx1+0.1)

    l0 = 0
    l1 = 0
    if self.n_values>1:
      l0 = idx0/(self.n_values-1)
      l1 = idx1/(self.n_values-1)

    hy0 = (1-l0)*self.y0 + l0*self.y1
    hy1 = (1-l1)*self.y0 + l1*self.y1

    self.bar.setRect( -1, hy0, 2, hy1-hy0 )

    if self.compose:
      self.label.widget().setStyleSheet('font-weight: bold;'+self.label._color)
    else:
      self.label.widget().setStyleSheet('font-weight: normal;'+self.label._color)

  def boundingRect(self):
    return self.br

  def paint(self, painter, option, widget):
    return

  def toSQL(self):
    if self.selection_idx1 < self.selection_idx0:
      idx0, idx1 = self.selection_idx1, self.selection_idx0
    else:
      idx0, idx1 = self.selection_idx0, self.selection_idx1

    v = []
    for i in range(idx0,idx1+1):
      v.append(self.values[i])
    v = tuple(v)

    if len(v)==1:
      return '"' + self.parameter + '" IN ("' + str(v[0]) + '")'
    else:
      return '"' + self.parameter + '" IN ' + str(v)

class SelectionEmitter(QtCore.QObject):
  s_selection_changed = QtCore.Signal(name='s_selection_changed')
  s_selection_changed_intermediate = QtCore.Signal(name='s_selection_changed_intermediate')

  def __init__(self):
    super().__init__()

class _ParallelCoordinatesView(QtWidgets.QGraphicsView):

  def __init__(self):
    super().__init__()
    self.axes = {}
    self.lines = Lines()
    self.emitter = SelectionEmitter()
    self.emitter.s_selection_changed_intermediate.connect(self.resize)

    self.header = []

    self.setRenderHints(QtGui.QPainter.Antialiasing)
    self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
    self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    self._scene = QtWidgets.QGraphicsScene(self)
    self._scene.addItem(self.lines)
    self.setScene(self._scene)

  def removeAxes(self):
    for a in self.axes:
      axis = self.axes[a]
      self._scene.removeItem(axis)
    self.axes = {}
    self.header = {}

  def addAxes(self,header,values):
    self.header = header

    for i, (p,idx) in enumerate(self.header):
      axis = Axis(p,values[i],self.emitter)
      self.axes[p] = axis
      self._scene.addItem(axis)

  def removeLines(self):
    self.lines.lines_normal = []
    self.lines.lines_highlight = []
    self.lines.path_normal = QtGui.QPainterPath()
    self.lines.path_highlight = QtGui.QPainterPath()

  def addLines(self, header, values, table, highlight=False):
    n = len(header) - 1
    if n<1: return

    if highlight:
      lines = self.lines.lines_highlight
    else:
      lines = self.lines.lines_normal

    dx = 1/n
    for row_idx in range(1,len(table)):
      path = []
      x = 0
      for header_idx, (p,column_idx) in enumerate(header):
        v = values[header_idx]
        vn = len(v) - 1
        if vn < 1:
          path.append( (x,0.5) )
        else:
          path.append( (x,v.index(table[row_idx][column_idx])/vn) )
        x += dx

      lines.append(path)

  def scrollContentsBy(self,a,b):
    return
  def wheelEvent(self, event):
    return
  def fitInView(self):
    return
  def keyPressEvent(self,event):
    return

  def resizeEvent(self, event):
    super().resizeEvent(event)
    self.resize()

  def resize(self):
    n = len(self.header)
    if n<1: return

    view_rect = self.mapToScene(self.rect()).boundingRect()

    margin = 50
    w = view_rect.width()-2*margin
    if n>1:
      x = view_rect.x()+margin
      aw = w/(n-1)
    else:
      x = view_rect.x() + view_rect.width()/2
      aw = w
    y = view_rect.y()
    h = view_rect.height()

    for (a,idx) in self.header:
      axis = self.axes[a]
      axis.update(x,y,h,aw)
      x+=aw

    a0 = self.axes[self.header[0][0]]
    a1 = self.axes[self.header[n-1][0]]
    x0 = a0.pos().x()
    x1 = a1.pos().x()
    y0 = a0.y0
    y1 = a0.y1
    self.lines.setBoundingRect(QtCore.QRectF(x0,y0,x1-x0,y1-y0))

  def applyState(self,state):
    for axis_name in state:
      axis = self.axes[axis_name]
      s = state[axis_name]
      axis.selection_idx0 = s[0]
      axis.selection_idx1 = s[1]
      axis.compose = s[2]

class ParallelCoordinatesView(Filter, FilterView):

    def __init__(self):
      FilterView.__init__(
        self,
        filter=self,
        delete_filter_on_close = True
      )

      Filter.__init__(
        self,
        inputs={
          'table': [[]],
          'ignore': ['^file','^id'],
          'state': {}
        },
        outputs={
          'table': [[]],
          'sql': 'SELECT * FROM input',
          'compose': (None,{})
        }
      )

    def generateWidgets(self):
      self.pcp = _ParallelCoordinatesView()
      self.content.layout().addWidget(self.pcp,1)
      self.pcp.emitter.s_selection_changed.connect(self.updateState)

    def updateState(self):
      state = {}
      for axis_name in self.pcp.axes:
        axis = self.pcp.axes[axis_name]
        if axis.selection_idx0 < axis.selection_idx1:
          idx0, idx1 = axis.selection_idx0, axis.selection_idx1
        else:
          idx1, idx0 = axis.selection_idx0, axis.selection_idx1
        state[axis_name] = (idx0,idx1,axis.compose)
      self.inputs.state.set(state)

    def _update(self):
      table = self.inputs.table.get()
      tableExtent = getTableExtent(table)
      self.pcp.removeAxes()
      self.pcp.removeLines()
      self.pcp.resize()
      if tableExtent[0]<2 or tableExtent[1]<1:
        self.outputs.table.set([[]])
        self.outputs.sql.set('SELECT * FROM input')
        return 1

      ignore = self.inputs.ignore.get()
      header = [(p,idx) for idx,p in enumerate(table[0]) if not any([re.search(i, p, re.IGNORECASE) for i in ignore])]
      header.sort()
      values = computeValues(header,table)

      self.pcp.addAxes(header,values)
      self.pcp.addLines(header,values,table)
      self.pcp.applyState(self.inputs.state.get())

      compose = (None,{})
      sql = 'SELECT * FROM input WHERE '
      for axis_name in self.pcp.axes:
        axis = self.pcp.axes[axis_name]
        axis_sql = axis.toSQL()
        if len(axis_sql)>0:
          sql += axis_sql+ ' AND '
        if axis.compose:
          valueMap = {}
          for i,v in enumerate(axis.values):
            valueMap[v] = i
          compose = (axis_name,valueMap)
      sql += ' '
      sql = sql[:-6]

      db = sqlite3.connect(":memory:")
      createTable(db, table)
      insertData(db, table)
      output_table = queryData(db, sql)

      header2 = [(p,output_table[0].index(p)) for (p,idx) in header]
      self.pcp.addLines( header2, values, output_table, True )

      self.pcp.resize()
      self.outputs.table.set(output_table)
      self.outputs.sql.set(sql)
      self.outputs.compose.set(compose)

      return 1
