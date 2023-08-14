from pycinema import isNumber, Filter, getTableExtent
from pycinema.theater.views.FilterView import FilterView

from PySide6 import QtCore, QtWidgets

class ParameterView(Filter, FilterView):

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
            'ignore': ['file','id','object_id_name'],
            'state': {}
          },
          outputs={
            'sql': 'SELECT * FROM input',
            'compose': (None,{})
          }
        )

    def generateWidgets(self):
        self.widgets = QtWidgets.QFrame()
        l = QtWidgets.QGridLayout()
        l.setAlignment(QtCore.Qt.AlignTop)
        l.setSpacing(0)
        l.setContentsMargins(0,0,0,0)
        self.widgets.setLayout(l)
        self.content.layout().addWidget(self.widgets)

        self.widgetsDict = {}

    def widgetToSQL(self, wt):
        v = None
        t = wt['T'].currentText()
        if t == 'S':
            v = wt['values'][wt['S'].value()]
        elif t == 'O':
            v = wt['O'].selectedItems()

        if type(v) is list:
            v = ['"'+i.text()+'"' for i in v]
            return '"' + wt['parameter'] + '" IN (' + ','.join(v) + ')'
        else:
            return '"' + wt['parameter'] + '"="' + v + '"'

    def computeValues(self,table,idx):
        vdic = set()

        for j in range(1,len(table)):
            vdic.add(table[j][idx])

        v_list = list(vdic)
        isListOfNumbers = isNumber(table[1][idx])
        if isListOfNumbers:
          v_list = [(float(x),x) for x in v_list]
        v_list.sort()
        if isListOfNumbers:
          v_list = [x[1] for x in v_list]

        return v_list

    def addWidgetToLayout(self, wt, grid_layout, grid_idx):
        wt['grid_idx'] = grid_idx
        grid_layout.addWidget(wt['C'],grid_idx,0,QtCore.Qt.AlignTop)
        grid_layout.addWidget(wt['T'],grid_idx,1,QtCore.Qt.AlignTop)
        grid_layout.addWidget(wt['SF'],grid_idx,2,QtCore.Qt.AlignTop)
        grid_layout.addWidget(wt['O'],grid_idx,2,QtCore.Qt.AlignTop)

    def generateWidget(self,parameter, table, idx, states):

        def on_change(name):
          self.update()

        def on_slider_change(name):
          wt = self.widgetsDict[name]
          wt['SL'].setText(wt['values'][wt['S'].value()])
          self.update()

        def on_type_change(name):
          wt = self.widgetsDict[name]

          t = wt['T'].currentText()
          if t == 'S':
              wt['SL'].setVisible(True)
              wt['O'].setVisible(False)
          elif t == 'O':
              wt['SL'].setVisible(False)
              wt['O'].setVisible(True)
          self.update()

        def make_callback(name,func):
            return lambda: func(name)

        wt = {}
        wt['parameter'] = parameter

        values = self.computeValues(table,idx)
        wt['values'] = values

        if parameter in states:
            state = states[parameter]
        else:
            state = {
              'C': False,
              'T': 'S',
              'S': 0,
              'O': [0],
            }

        wt['C'] = QtWidgets.QPushButton(parameter)
        wt['C'].setCheckable(True)
        wt['C'].setChecked(state['C'])
        wt['C'].toggled.connect( make_callback(parameter, on_change ) )

        wt['T'] = QtWidgets.QComboBox()
        wt['T'].addItems(["S", "O"])
        wt['T'].setCurrentText(state['T'])
        wt['T'].currentTextChanged.connect( make_callback(parameter, on_type_change ) )

        wt['S'] = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        wt['S'].setMinimum(0)
        wt['S'].setMaximum(len(values)-1)
        wt['S'].setValue(state['S'])
        wt['S'].valueChanged.connect( make_callback(parameter, on_slider_change ) )

        wt['SL'] = QtWidgets.QLabel(values[state['S']])
        wt['SL'].setVisible(state['T']=='S')

        wt['SF'] = QtWidgets.QFrame()
        wt['SF'].layout = QtWidgets.QHBoxLayout(wt['SF'])
        wt['SF'].layout.addWidget(wt['S'])
        wt['SF'].layout.addWidget(wt['SL'])

        wt['O'] = QtWidgets.QListWidget()
        wt['O'].insertItems(0,values)
        wt['O'].setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        wt['O'].setMaximumHeight(60)
        wt['O'].setVisible(state['T']=='O')
        for oidx in state['O']:
            wt['O'].item(oidx).setSelected(True)
        wt['O'].itemSelectionChanged.connect( make_callback(parameter, on_change ) )

        return wt

    def updateWidgets(self):

        table = self.inputs.table.get()
        states = self.inputs.state.get()
        ignore = self.inputs.ignore.get()

        parameters = [p for p in table[0] if p not in ignore]
        parameters.sort()

        existing_parameters = [p for p in self.widgetsDict]
        existing_parameters.sort()

        if parameters != existing_parameters and len(existing_parameters)<1:
          grid_layout = self.widgets.layout()
          grid_idx = 0
          for parameter in parameters:
              idx = table[0].index(parameter)
              wt = self.generateWidget(parameter,table,idx,states)
              self.widgetsDict[parameter] = wt
              self.addWidgetToLayout(wt,grid_layout,grid_idx)
              grid_idx += 1

    def _update(self):

        table = self.inputs.table.get()
        tableExtent = getTableExtent(table)
        if tableExtent[0]<1 or tableExtent[1]<1:
          self.outputs.sql.set('')
          self.outputs.compose.set((None,{}))
          return 0

        # compute widgets
        self.updateWidgets()

        # export state
        state = {}
        for _,wt in self.widgetsDict.items():
            state[wt['parameter']] = {
              "C": wt['C'].isChecked(),
              "T": wt['T'].currentText(),
              "S": wt['S'].value(),
              "O": [wt['O'].indexFromItem(i).row() for i in wt['O'].selectedItems()]
            }
        self.inputs.state.set(state, False)

        sql = 'SELECT * FROM input WHERE '
        for _,wt in self.widgetsDict.items():
            wsql = self.widgetToSQL(wt)
            if len(wsql)>0:
                sql += wsql+ ' AND '

        sql += ' '

        self.outputs.sql.set(sql[:-6])

        compose = (None,{})
        for _,wt in self.widgetsDict.items():
            if wt['C'].isChecked():
                valueMap = {}
                for i,v in enumerate(wt['values']):
                    valueMap[v] = i
                compose = (wt['parameter'],valueMap)
                break
        self.outputs.compose.set(compose)

        return 1
