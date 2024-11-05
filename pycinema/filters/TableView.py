from pycinema.filters.ImagesToTable import ImagesToTable
from pycinema import Filter, Image

try:
  from PySide6 import QtGui, QtCore, QtWidgets
except ImportError:
  pass

try:
  class TableModel(QtCore.QAbstractTableModel):
      def __init__(self):
          super().__init__()
          self._data = []

      def setData(self,data):
          self._data = data
          self.modelReset.emit()

      def data(self, index, role):
          if role == QtCore.Qt.DisplayRole:
              return self._data[index.row()+1][index.column()]

      def rowCount(self, index):
          return len(self._data)-1

      def columnCount(self, index):
          if len(self._data)>0:
              return len(self._data[0])
          else:
              return 0

      def headerData(self, section, orientation, role = QtCore.Qt.DisplayRole):
          if role == QtCore.Qt.DisplayRole and orientation==QtCore.Qt.Horizontal:
              return self._data[0][section]
          else:
              return super().headerData(section,orientation,role)

  class NumericSortProxyModel(QtCore.QSortFilterProxyModel):
      def lessThan(self, left, right):
          leftData = self.sourceModel().data(left, QtCore.Qt.DisplayRole)
          rightData = self.sourceModel().data(right, QtCore.Qt.DisplayRole)

          try:
              leftValue = float(leftData)
              rightValue = float(rightData)
              return leftValue > rightValue
          except ValueError:
              return leftData > rightData

except NameError:
  pass

class TableView(Filter):

    def __init__(self):

        self.model = TableModel()

        self.proxyModel = NumericSortProxyModel()
        self.proxyModel.setSourceModel(self.model)

        self.selection_model = QtCore.QItemSelectionModel()
        self.selection_model.setModel(self.proxyModel)
        self.selection_model.selectionChanged.connect(self._onSelectionChanged)

        self.suppress_selection_update = False

        self.outputTable = list()

        self.widgets = []
        self.table_input_time = -1

        Filter.__init__(
          self,
          inputs={
            'table': [[]],
            'selection': []
          },
          outputs={
            'table': [[]]
          }
        )

    def generateWidgets(self):
        widget = QtWidgets.QTableView()
        widget.setModel(self.proxyModel)
        widget.setSortingEnabled(True)
        widget.setSelectionModel(self.selection_model)
        widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.widgets.append(widget)
        return widget

    def _onSelectionChanged(self, selected, deselected):
        if self.suppress_selection_update:
            return
        indices = set(self.proxyModel.mapToSource(index).row() for index in self.selection_model.selectedIndexes())

        table = self.inputs.table.get()
        input_is_image_list = len(table)>0 and isinstance(table[0],Image)
        selection = None
        if input_is_image_list:
          selection = [table[i].meta['id'] for i in indices]
        else:
          id_column_idx = table[0].index('id')
          selection = [table[i+1][id_column_idx] for i in indices]

        self.inputs.selection.set(selection,True,True)

    def _update(self):
        table = self.inputs.table.get()
        input_is_image_list = len(table)>0 and isinstance(table[0],Image)

        if self.table_input_time != self.inputs.table.getTime():
          self.table_input_time = self.inputs.table.getTime()
          table_data = table
          if input_is_image_list:
            table_data = ImagesToTable.imagesToTable(table)
          self.model.setData(table_data)

        selection = self.inputs.selection.get()
        selection_indices = []
        output_table = [[]]
        id_column_idx = 0

        selection_mode = QtWidgets.QAbstractItemView.ExtendedSelection

        if input_is_image_list:
          selection_indices = [i for i in range(0,len(table)) if table[i].meta['id'] in selection]
          output_table = [table[i] for i in selection_indices]
        else:
          try: id_column_idx = table[0].index('id')
          except ValueError: id_column_idx = -1

          if id_column_idx<0:
            selection_mode = QtWidgets.QAbstractItemView.NoSelection
          else:
            selection_indices = [i for i in range(0,len(table)) if table[i][id_column_idx] in selection]
            output_table = [table[i] for i in [0]+selection_indices]
            selection_indices = [i-1 for i in selection_indices]

        # disable selection if no id column present
        for w in self.widgets:
            w.setSelectionMode(selection_mode)

        self.outputs.table.set( output_table )

        self.suppress_selection_update = True
        self.selection_model.clear()
        if id_column_idx<0:
          self.inputs.selection.set([])
        else:
          indices_ = [self.model.index(r, 0) for r in selection_indices]
          mode = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
          [self.selection_model.select(self.proxyModel.mapFromSource(i), mode) for i in indices_]
        self.suppress_selection_update = False

        return 1
