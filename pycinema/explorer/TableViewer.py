from pycinema import Filter

import numpy
# import PIL
from PySide6 import QtCore, QtWidgets, QtGui

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

class TableViewer(Filter):

    def __init__(self):
        self.model = TableModel()
        self.tableView = QtWidgets.QTableView()
        self.tableView.setModel(self.model)

        super().__init__(
          inputs={
            'table': [[]],
            'container': None
          }
        )

    def _update(self):

        container = self.inputs.container.get()
        if container and container != self.tableView.parent():
            container.layout().addWidget(self.tableView,1)

        table = self.inputs.table.get()
        self.model.setData(table)

        return 1
