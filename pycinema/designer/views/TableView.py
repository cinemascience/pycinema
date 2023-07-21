from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.designer.views.FilterView import FilterView
from pycinema import Filter

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

        FilterView.__init__(
          self,
          filter=self,
          delete_filter_on_close = True
        )

        Filter.__init__(
          self,
          inputs={
            'table': [[]]
          }
        )

    def generateWidgets(self):
        self.tableView = QtWidgets.QTableView()
        # self.tableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch);
        # self.tableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive);
        # self.tableView.horizontalHeader().resizeSections(QtWidgets.QHeaderView.Stretch);
        self.tableView.setModel(self.model)
        self.content.layout().addWidget(self.tableView,1)

    def _update(self):

        table = self.inputs.table.get()
        self.model.setData(table)

        return 1
