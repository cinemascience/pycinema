from pycinema import isNumber
from .FilterView import ViewFilter

from PySide6 import QtCore, QtWidgets

from pycinema import getTableExtent

class ParameterViewer(ViewFilter):

    def __init__(self, view):

        self.widgets = QtWidgets.QFrame()
        self.widgets.setLayout(QtWidgets.QGridLayout())
        self.widgets.layout().setAlignment(QtCore.Qt.AlignTop) 
        self.widgets.layout().setSpacing(0)
        self.widgets.layout().setContentsMargins(0,0,0,0)
        # self.widgets.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Raised)
        view.content.layout().addWidget(self.widgets)

        self.widgets_ = {}

        super().__init__(
            inputs={
              'table': [[]],
              'ignore': ['file','id','object_id_name'],
              'state': {}
            },
            outputs={
              'sql': 'SELECT * FROM input',
              'composite_by_meta': (None,{})
            }
        )

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

    def generateWidgets(self):

        table = self.inputs.table.get()

        gridL = self.widgets.layout()

        def on_change(name):
          self.update()

        def on_slider_change(name):
          wt = self.widgets_[name]
          wt['SL'].setText(wt['values'][wt['S'].value()])
          self.update()

        def on_type_change(name):
          wt = self.widgets_[name]

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

        states = self.inputs.state.get()

        header = table[0]
        for i in range(0,len(header)):
            parameter = header[i]
            if parameter in self.inputs.ignore.get():
                continue

            vdic = set()

            for j in range(1,len(table)):
                vdic.add(table[j][i])

            values = list(vdic)
            isListOfNumbers = isNumber(table[1][i])
            if isListOfNumbers:
                values = [float(x) for x in values]
            values.sort()
            if isListOfNumbers:
                values = [str(x) for x in values]

            wt = {}
            wt['parameter'] = parameter
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
            gridL.addWidget(wt['C'],i,0,QtCore.Qt.AlignTop) 

            wt['T'] = QtWidgets.QComboBox()
            wt['T'].addItems(["S", "O"])
            wt['T'].setCurrentText(state['T'])
            gridL.addWidget(wt['T'],i,1, QtCore.Qt.AlignTop)
            wt['T'].currentTextChanged.connect( make_callback(parameter, on_type_change ) )

            wt['S'] = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            wt['S'].setMinimum(0)
            wt['S'].setMaximum(len(values)-1)
            wt['S'].setValue(state['S'])

            wt['SL'] = QtWidgets.QLabel(values[state['S']])
            wt['SL'].setVisible(state['T']=='S')
            wt['SL'].setFixedWidth(50)
            wt['SL'].setAlignment(QtCore.Qt.AlignRight)
            wt['S'].valueChanged.connect( make_callback(parameter, on_slider_change ) )
            # slider frame
            wt['SF'] = QtWidgets.QFrame()
            # wt['SF'].setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Raised)
            wt['SF'].layout = QtWidgets.QHBoxLayout(wt['SF'])
            wt['SF'].layout.setAlignment(QtCore.Qt.AlignTop) 
            wt['SF'].layout.setSpacing(0)
            wt['SF'].layout.setContentsMargins(0,0,0,0)
            wt['SF'].layout.addWidget(wt['S'], QtCore.Qt.AlignTop)
            wt['SF'].layout.addWidget(wt['SL'], QtCore.Qt.AlignTop)
            gridL.addWidget(wt['SF'],i,2, QtCore.Qt.AlignTop)

            wt['O'] = QtWidgets.QListWidget()
            wt['O'].insertItems(0,values)
            wt['O'].setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            wt['O'].setMaximumHeight(60)
            wt['O'].setVisible(state['T']=='O')
            for oidx in state['O']:
                wt['O'].item(oidx).setSelected(True)

            wt['O'].itemSelectionChanged.connect( make_callback(parameter, on_change ) )
            gridL.addWidget(wt['O'],i,2,QtCore.Qt.AlignTop)

            self.widgets_[parameter] = wt

    def _update(self):

        table = self.inputs.table.get()
        tableExtent = getTableExtent(table)
        if tableExtent[0]<1 or tableExtent[1]<1:
          self.outputs.sql.set('')
          self.outputs.composite_by_meta.set((None,{}))
          return 0

        sql = 'SELECT * FROM input WHERE '

        # compute widgets
        if len(self.widgets_) < 1:
            self.generateWidgets()
            spacer = QtWidgets.QLabel("")
            spacer.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
            self.widgets.layout().addWidget(spacer)

        # export state
        state = {}
        for _,wt in self.widgets_.items():
            state[wt['parameter']] = {
              "C": wt['C'].isChecked(),
              "T": wt['T'].currentText(),
              "S": wt['S'].value(),
              "O": [wt['O'].indexFromItem(i).row() for i in wt['O'].selectedItems()]
            }
        self.inputs.state.set(state, False)

        for _,wt in self.widgets_.items():
            wsql = self.widgetToSQL(wt)
            if len(wsql)>0:
                sql += wsql+ ' AND '

        sql += ' '

        self.outputs.sql.set(sql[:-6])

        composite_by_meta = (None,{})
        for _,wt in self.widgets_.items():
            if wt['C'].isChecked():
                valueMap = {}
                for i,v in enumerate(wt['values']):
                    valueMap[v] = i
                composite_by_meta = (wt['parameter'],valueMap)
                break
        self.outputs.composite_by_meta.set(composite_by_meta)

        return 1
