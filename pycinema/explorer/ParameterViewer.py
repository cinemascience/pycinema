from pycinema import Filter, isNumber

from PySide6 import QtCore, QtWidgets

class ParameterViewer(Filter):

    def __init__(self):
        self.widgets = []

        super().__init__(
            inputs={
              'table': [[]],
              'container': None,
              'ignore': ['file','id','object_id_name']
            },
            outputs={
              'sql': 'SELECT * FROM input',
              'composite_by_meta': (None,{})
            }
        )

    def widgetToSQL(self, wt):
        v = None
        match wt['T'].currentText():
            case 'S':
                v = wt['values'][wt['S'].value()]
            case 'O':
                v = wt['O'].selectedItems()

        if type(v) is list:
            v = ['"'+i.text()+'"' for i in v]
            return '"' + wt['parameter'] + '" IN (' + ','.join(v) + ')'
        else:
            return '"' + wt['parameter'] + '"="' + v + '"'

    def generateWidgets(self):

        table = self.inputs.table.get()
        self.widgets = []

        container = self.inputs.container.get()

        grid = QtWidgets.QFrame()
        grid.setLayout(QtWidgets.QGridLayout())
        gridL = grid.layout()

        container.layout().addWidget(grid)
        container.layout().addWidget(QtWidgets.QLabel(""),1)

        def on_change(idx):
          self.update()

        def on_slider_change(idx):
          wt = self.widgets[idx]
          wt['SL'].setText(wt['values'][wt['S'].value()])
          self.update()

        def on_type_change(idx):
          wt = self.widgets[idx]

          match wt['T'].currentText():
            case 'S':
              wt['SL'].setVisible(True)
              wt['O'].setVisible(False)
            case 'O':
              wt['SL'].setVisible(False)
              wt['O'].setVisible(True)
          self.update()

        def make_callback(idx,func):
            return lambda: func(idx)

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

            wt['C'] = QtWidgets.QPushButton(parameter)
            wt['C'].setCheckable(True)
            wt['C'].toggled.connect( make_callback(len(self.widgets), on_change ) )
            gridL.addWidget(wt['C'],i,0)

            wt['T'] = QtWidgets.QComboBox()
            wt['T'].addItems(["S", "O"])
            gridL.addWidget(wt['T'],i,1)
            wt['T'].currentTextChanged.connect( make_callback(len(self.widgets), on_type_change ) )

            wt['S'] = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            wt['S'].setMinimum(0)
            wt['S'].setMaximum(len(values)-1)
            wt['S'].setValue(0)

            # SL = QtWidgets.QLabel("TODO")
            wt['SL'] = QtWidgets.QLabel(values[0])
            wt['S'].valueChanged.connect( make_callback(len(self.widgets), on_slider_change ) )
            wt['SF'] = QtWidgets.QFrame()
            wt['SF'].layout = QtWidgets.QHBoxLayout(wt['SF'])
            wt['SF'].layout.addWidget(wt['S'])
            wt['SF'].layout.addWidget(wt['SL'])
            gridL.addWidget(wt['SF'],i,2)

            wt['O'] = QtWidgets.QListWidget()
            wt['O'].insertItems(0,values)
            wt['O'].setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            wt['O'].setMaximumHeight(60)
            wt['O'].setCurrentRow(0)
            wt['O'].setVisible(False)
            wt['O'].itemSelectionChanged.connect( make_callback(len(self.widgets), on_change ) )
            gridL.addWidget(wt['O'],i,2)

            self.widgets.append(wt)

    def _update(self):

        table = self.inputs.table.get()
        if not table or len(table)<1:
          self.outputs.sql.set("")
          self.outputs.composite_by_meta.set((None,{}))
          return

        sql = 'SELECT * FROM input WHERE '

        # compute widgets
        if len(self.widgets) < 1:
            self.generateWidgets()

        for i,wt in enumerate(self.widgets):
            wsql = self.widgetToSQL(wt)
            if len(wsql)>0:
                sql += wsql+ ' AND '

        sql += ' '

        self.outputs.sql.set(sql[:-6])

        composite_by_meta = (None,{})
        for i,wt in enumerate(self.widgets):
            if wt['C'].isChecked():
                valueMap = {}
                for i,v in enumerate(wt['values']):
                    valueMap[v] = i
                composite_by_meta = (wt['parameter'],valueMap)
                break
        self.outputs.composite_by_meta.set(composite_by_meta)

        return 1
