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

except NameError:
  pass

class TableView(Filter):

    def __init__(self):

        self.model = TableModel()
        self.selection_model = QtCore.QItemSelectionModel()
        self.selection_model.setModel(self.model)
        self.selection_model.selectionChanged.connect(self._onSelectionChanged)

        self.update_from_selection = False
        self.suppress_selection_update = False

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
        widget.setModel(self.model)
        widget.setSelectionModel(self.selection_model)
        widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        return widget

    def _onSelectionChanged(self, selected, deselected):
        if self.suppress_selection_update:
            return
        self.update_from_selection = True
        indices = list(set(index.row() for index in self.selection_model.selectedIndexes()))
        indices.sort()
        if self.inputs.selection.valueIsPort():
          self.inputs.selection._value.parent.inputs.value.set(indices)
        else:
          self.inputs.selection.set(indices)

    def _update(self):
        table = self.inputs.table.get()
        input_is_image_list = len(table)>0 and isinstance(table[0],Image)

        if not self.update_from_selection:
            table_data = table
            if input_is_image_list:
                table_data = ImagesToTable.imagesToTable(table)
            self.model.setData(table_data)
        self.update_from_selection = False

        indices = list(self.inputs.selection.get())

        if input_is_image_list:
            self.outputs.table.set(
                [table[i] for i in indices if i<len(table)]
            )
        else:
            indices.insert(0,-1)
            self.outputs.table.set(
                [table[i+1] for i in indices  if i+1<len(table)]
            )

        self.suppress_selection_update = True
        indices_ = [self.model.index(r, 0) for r in indices]
        mode = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        [self.selection_model.select(i, mode) for i in indices_]
        self.suppress_selection_update = False

        return 1
