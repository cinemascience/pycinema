from copy import deepcopy
import re
import sqlite3

from pycinema import isNumber, Filter, getTableExtent
from pycinema.filters.TableQuery import executeSQL, createTable, insertData, queryData

try:
  from PySide6 import QtGui, QtCore, QtWidgets
  from pycinema.theater.Icons import Icons
except Exception:
  pass

try:
  class Emitter(QtCore.QObject):
    s_update = QtCore.Signal()
    def __init__(self):
      super().__init__()
except NameError:
  class Emitter():
    def __init__(self):
      pass

def computeValues(table,idx):
  vdic = set()

  for j in range(1,len(table)):
      vdic.add(table[j][idx])

  v_list = list(vdic)
  isListOfNumbers = isNumber(table[1][idx])
  if isListOfNumbers:
    v_list = [(float(x),x) for x in v_list]
  v_list.sort()
  if isListOfNumbers:
    v_list = [str(x[1]) for x in v_list]

  return v_list

# ==============================================================================
# ParametersView
# ==============================================================================
class ParametersView(Filter):

    def __init__(self):
        self.emitter = Emitter()
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

    def updateWidgets(self,widgets):
      state = self.inputs.state.get()

      def update_mode(p,v):
        if self.ignore: return
        new_state = deepcopy(self.inputs.state.get())
        if v:
          new_state[p]['M'] = 'O'
        else:
          new_state[p]['M'] = 'S'
          vv = new_state[p]['V']
          if len(vv):
            new_state[p]['V'] = [vv[0]]
          else:
            new_state[p]['V'] = [new_state[p]['O'][0]]

        self.inputs.state.set(new_state)


      def update_compositing(p,v):
        if self.ignore: return
        new_state = deepcopy(self.inputs.state.get())
        new_state[p]['C'] = v
        self.inputs.state.set(new_state)
      def update_slider(p,v):
        if self.ignore: return
        new_state = deepcopy(self.inputs.state.get())
        new_state[p]['V'] = [v]
        self.inputs.state.set(new_state)
      def update_options(p,o):
        if self.ignore: return
        new_state = deepcopy(self.inputs.state.get())
        new_state[p]['V'] = [o.indexFromItem(i).row() for i in o.selectedItems()]
        self.inputs.state.set(new_state)

      def make_callback(p,func):
        return lambda v: func(p,v)
      def make_callback2(p,o,func):
        return lambda: func(p,o)

      # check if generation is necessary
      requires_generation = any([s not in widgets.child_dict for s in state])
      if requires_generation:
        widgets.container.setParent(None)
        widgets.container.deleteLater()
        widgets.container = QtWidgets.QFrame()
        widgets.layout().addWidget(widgets.container)

        layout = QtWidgets.QGridLayout()
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        widgets.container.setLayout(layout)
        widgets.child_dict = {}

        row_idx = 0
        for p in state:
          state_ = state[p]
          l = QtWidgets.QPushButton(p)
          l.setCheckable(True)
          l.toggled.connect( make_callback(p,update_compositing) )
          layout.addWidget(l,row_idx,0,QtCore.Qt.AlignVCenter)

          m = QtWidgets.QPushButton()
          m.setCheckable(True)
          m.setIcon( Icons.toQIcon(Icons.icon_list) )
          m.setCursor(QtCore.Qt.PointingHandCursor)
          m.setToolTip('Toggle slider/list widget')
          m.toggled.connect( make_callback(p,update_mode) )
          layout.addWidget(m,row_idx,1,QtCore.Qt.AlignVCenter)

          s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
          s.setMinimum(0)
          s.setMaximum(len(state_['O'])-1)
          s.valueChanged.connect( make_callback(p,update_slider) )

          v = QtWidgets.QLabel()
          mwidth = v.fontMetrics().boundingRect('m').width()
          maxWidth = max(len(str(x)) for x in state_['O'])
          v.setMinimumWidth(maxWidth*mwidth)

          sv = QtWidgets.QFrame()
          sv.setLayout(QtWidgets.QHBoxLayout())
          sv.layout().addWidget(s)
          sv.layout().addWidget(v)
          layout.addWidget(sv,row_idx,2,QtCore.Qt.AlignVCenter)

          o = QtWidgets.QListWidget()
          o.insertItems(0,state_['O'])
          o.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
          # o.setMaximumHeight(1.2*mwidth*min(4,len(state_['O'])))
          layout.addWidget(o,row_idx,2,QtCore.Qt.AlignVCenter)
          o.itemSelectionChanged.connect( make_callback2(p, o, update_options ) )

          widgets.child_dict[p] = {
            'l': l,
            'm': m,
            's': s,
            'v': v,
            'o': o,
            'sv': sv
          }
          row_idx += 1

      self.ignore = True
      for p in state:
        state_ = state[p]
        child = widgets.child_dict[p]

        child['l'].setChecked(state_['C'])
        child['m'].setChecked(state_['M']!='S')
        if len(state_['V']):
          child['s'].setValue(state_['V'][0])
        child['v'].setText(state_['O'][state_['V'][0]])
        child['sv'].setVisible(state_['M']=='S')
        child['o'].setVisible(state_['M']=='O')
        child['o'].clearSelection()
        for x in state_['V']:
          child['o'].item(x).setSelected(True)
      self.ignore = False
      return

    def generateWidgets(self):
        widgets = QtWidgets.QFrame()
        widgets.setLayout(QtWidgets.QHBoxLayout())
        widgets.container = QtWidgets.QWidget()
        widgets.layout().addWidget(widgets.container)
        widgets.child_dict = {}

        self.emitter.s_update.connect(lambda: self.updateWidgets(widgets))
        self.updateWidgets(widgets)
        return widgets

    def _update(self):
        table = self.inputs.table.get()
        tableExtent = getTableExtent(table)
        if tableExtent[0]<1 or tableExtent[1]<1:
          self.outputs.table.set([[]])
          self.outputs.sql.set('')
          self.outputs.compose.set((None,{}))
          return 0

        state = self.inputs.state.get()
        if not bool(state):
          state = {}
          ignore = self.inputs.ignore.get()
          parameterIndices = [idx for idx in range(0,tableExtent[1]) if not any([re.search(i, table[0][idx], re.IGNORECASE) for i in ignore])]
          for i in parameterIndices:
            state[table[0][i]] = {
              'C': False,
              'O': computeValues(table,i),
              'V': [0],
              'M': 'S'
            }
          self.inputs.state.set(state)

        # sql output
        sql = 'SELECT * FROM input WHERE '
        for s in state:
          sql += '"'+s+'" IN ("' +'","'.join([str(state[s]['O'][x]) for x in state[s]['V']])+ '") AND '

        sql += ' '
        sql = sql[:-6]
        self.outputs.sql.set(sql)

        # table output
        db = sqlite3.connect(":memory:")
        createTable(db, table)
        insertData(db, table)
        output_table = queryData(db, sql)
        self.outputs.table.set(output_table)

        # compositing output
        compose = [(s, {v:i for i,v in enumerate(state[s]['O'])} ) for s in state if state[s]['C']]
        if len(compose)<1:
          compose = (None,{})
        else:
          compose = compose[0]
        self.outputs.compose.set(compose)

        self.emitter.s_update.emit()

        return 1
