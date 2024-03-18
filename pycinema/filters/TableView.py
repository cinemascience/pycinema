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

class TableView(Filter):

    def __init__(self):

        self.model = TableModel()

        self.proxyModel = NumericSortProxyModel() #QtCore.QSortFilterProxyModel()
        self.proxyModel.setSourceModel(self.model)


        self.selection_model = QtCore.QItemSelectionModel(self.proxyModel)
        #self.selection_model.setModel(self.model)
        self.selection_model.selectionChanged.connect(self._onSelectionChanged)

        self.update_from_selection = False
        self.suppress_selection_update = False

        self.outputTable = list()

        Filter.__init__(
          self,
          inputs={
            'table': [[]],
            'selection': []
          },
          outputs={
            'tableSelection': [[]],
            'table': [[]]
          }
        )

    def generateWidgets(self):
        widget = QtWidgets.QTableView()
        widget.setModel(self.model)
        widget.setModel(self.proxyModel)
        widget.setSortingEnabled(True)
        widget.horizontalHeader().sectionClicked.connect(self._onHeaderClicked)        
        widget.setSelectionModel(self.selection_model)
        widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        return widget

    def _onSelectionChanged(self, selected, deselected):
        if self.suppress_selection_update:
            return
        self.update_from_selection = True
        indices = list(set(index.row() for index in self.selection_model.selectedIndexes()))
        indices.sort()
        self.inputs.selection.set(indices,True,True)

    def _onHeaderClicked(self, logicalIndex):
        # reorder the output table
        self.outputTable = list()
        rowCount = self.proxyModel.rowCount()
        columnCount = self.proxyModel.columnCount()

        # add header info
        self.outputTable.append(self.inputs.table.get()[0])
        for row in range(rowCount):
            rowData = []
            for column in range(columnCount):
                index = self.proxyModel.index(row, column)
                data = self.proxyModel.data(index, QtCore.Qt.DisplayRole)
                rowData.append(data)
            self.outputTable.append(tuple(rowData))

        # update indices of selected rows
        self._onSelectionChanged(None, None)
        # push to update
        self.update()


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
            self.outputs.tableSelection.set(
                [table[i] for i in indices if i<len(table)]
            )
        else:
            indices.insert(0,-1)
            self.outputs.tableSelection.set(
                [table[i+1] for i in indices  if i+1<len(table)]
            )

        self.suppress_selection_update = True
        indices_ = [self.model.index(r, 0) for r in indices]
        mode = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        [self.selection_model.select(i, mode) for i in indices_]
        self.suppress_selection_update = False

        # list empty, ie. no header clicked, then use input as output
        if len(self.outputTable) == 0: 
            self.outputs.table.set(list(self.inputs.table.get()))
        # else, pull order of rows from internal variable
        else:
            self.outputs.table.set(self.outputTable)

        return 1
