from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.views.FilterView import FilterView
from pycinema.filters.ImagesToTable import ImagesToTable
from pycinema import Filter, Image

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

class TableView(Filter, FilterView):

    def __init__(self):

        self.model = TableModel()
        self.update_from_selection = False
        self.suppress_selection_update = False

        FilterView.__init__(
          self,
          filter=self,
          delete_filter_on_close = True
        )

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
        self.tableView = QtWidgets.QTableView(self)
        # self.tableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch);
        # self.tableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive);
        # self.tableView.horizontalHeader().resizeSections(QtWidgets.QHeaderView.Stretch);
        self.tableView.setModel(self.model)
        self.tableView.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableView.selectionModel().selectionChanged.connect(self._onSelectionChanged)
        self.content.layout().addWidget(self.tableView,1)

    def _onSelectionChanged(self, selected, deselected):
        if self.suppress_selection_update:
            return
        self.update_from_selection = True
        indices = list(set(index.row() for index in self.tableView.selectedIndexes()))
        indices.sort()
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
                [table[i] for i in indices]
            )
        else:
            indices.insert(0,-1)
            self.outputs.table.set(
                [table[i+1] for i in indices]
            )

        self.suppress_selection_update = True
        selection_model = self.tableView.selectionModel()
        indices_ = [self.tableView.model().index(r, 0) for r in indices]
        mode = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        [self.tableView.selectionModel().select(i, mode) for i in indices_]
        self.suppress_selection_update = False

        return 1
