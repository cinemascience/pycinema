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

# class NumericSortProxyModel(QtCore.QSortFilterProxyModel):
#     def lessThan(self, left, right):
#         leftData = self.sourceModel().data(left, QtCore.Qt.DisplayRole)
#         rightData = self.sourceModel().data(right, QtCore.Qt.DisplayRole)

#         try:
#             leftValue = float(leftData)
#             rightValue = float(rightData)
#             return leftValue > rightValue
#         except ValueError:
#             return leftData > rightData

class TableView(Filter):

    def __init__(self):

        self.model = TableModel()

        # self.proxyModel = NumericSortProxyModel() #QtCore.QSortFilterProxyModel()
        # self.proxyModel.setSourceModel(self.model)
        self.selection_model = QtCore.QItemSelectionModel() #self.proxyModel
        self.selection_model.setModel(self.model)
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
            'table': [[]]
          }
        )

    def generateWidgets(self):
        widget = QtWidgets.QTableView()
        widget.setModel(self.model)
        #widget.setModel(self.proxyModel)
        #widget.setSortingEnabled(True)
        #widget.horizontalHeader().sectionClicked.connect(self._onHeaderClicked)
        widget.setSelectionModel(self.selection_model)
        widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        return widget

    def _onSelectionChanged(self, selected, deselected):
        if self.suppress_selection_update:
            return
        self.update_from_selection = True
        indices = set(index.row() for index in self.selection_model.selectedIndexes())

        table = self.inputs.table.get()
        input_is_image_list = len(table)>0 and isinstance(table[0],Image)
        selection = None
        if input_is_image_list:
          selection = [table[i].meta['id'] for i in indices]
        else:
          id_column_idx = table[0].index('id')
          selection = [table[i+1][id_column_idx] for i in indices]

        self.inputs.selection.set(selection,True,True)

    # def _onHeaderClicked(self, logicalIndex):
    #     # reorder the output table
    #     self.outputTable = list()
    #     rowCount = self.proxyModel.rowCount()
    #     columnCount = self.proxyModel.columnCount()

    #     # add header info
    #     self.outputTable.append(self.inputs.table.get()[0])
    #     for row in range(rowCount):
    #         rowData = []
    #         for column in range(columnCount):
    #             index = self.proxyModel.index(row, column)
    #             data = self.proxyModel.data(index, QtCore.Qt.DisplayRole)
    #             rowData.append(data)
    #         self.outputTable.append(tuple(rowData))

    #     # update indices of selected rows
    #     self._onSelectionChanged(None, None)
    #     # push to update
    #     self.update()


    def _update(self):
        table = self.inputs.table.get()
        input_is_image_list = len(table)>0 and isinstance(table[0],Image)

        if not self.update_from_selection:
            table_data = table
            if input_is_image_list:
                table_data = ImagesToTable.imagesToTable(table)
            self.model.setData(table_data)
        self.update_from_selection = False

        selection = self.inputs.selection.get()
        selection_indices = None
        output_table = None
        if input_is_image_list:
          selection_indices = [i for i in range(0,len(table)) if table[0].meta['id'] in selection]
          output_table = [table[i] for i in selection_indices]
        else:
          id_column_idx = table[0].index('id')
          selection_indices = [i for i in range(0,len(table)) if table[i][id_column_idx] in selection]
          output_table = [table[0]]
          for i in selection_indices:
            output_table.append(table[i])

        self.outputs.table.set( output_table )

        self.suppress_selection_update = True
        indices_ = [self.model.index(r-1, 0) for r in selection_indices]
        mode = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        [self.selection_model.select(i, mode) for i in indices_]
        self.suppress_selection_update = False

        # # list empty, ie. no header clicked, then use input as output
        # if len(self.outputTable) == 0:
        #     self.outputs.table.set(list(self.inputs.table.get()))
        # # else, pull order of rows from internal variable
        # else:
        #     self.outputs.table.set(self.outputTable)

        return 1
