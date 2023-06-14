from PySide6 import QtCore, QtWidgets, QtGui

import pycinema
import pycinema.filters

class QueryWidget(QtWidgets.QLineEdit):

    s_up = QtCore.Signal(name='up')
    s_down = QtCore.Signal(name='down')

    def __init__(self):
        super().__init__()

    def keyReleaseEvent(self,event):
        super().keyReleaseEvent(event)
        if event.key()==QtCore.Qt.Key_Up:
            self.s_up.emit()
        elif event.key()==QtCore.Qt.Key_Down:
            self.s_down.emit()

class FilterBrowser(QtWidgets.QDialog):

    filters = dict([(name, cls) for name, cls in pycinema.filters.__dict__.items() if isinstance(cls,type) and issubclass(cls,pycinema.Core.Filter)])

    def __init__(self):
        super().__init__()

        self.setLayout(QtWidgets.QVBoxLayout())

        self.query = QueryWidget()
        self.query.textChanged.connect(self.updateList)
        self.query.s_up.connect(lambda: self.updateSelection(-1))
        self.query.s_down.connect(lambda: self.updateSelection(1))
        self.query.returnPressed.connect(self.submit)

        self.layout().addWidget(self.query)

        self.hiddenList = QtWidgets.QListWidget()

        for name in self.filters:
            self.hiddenList.addItem(name)
        self.hiddenList.sortItems()

        self.list = QtWidgets.QListWidget()
        self.updateList()

        self.layout().addWidget(self.list,1)

    def updateList(self,query=""):
        items = self.hiddenList.findItems(query, QtCore.Qt.MatchContains)
        for i in range(self.list.count(),-1,-1):
            self.list.takeItem(i)
        for i in items:
            self.list.addItem(i.text())
        self.list.setCurrentRow(0)

    def updateSelection(self,direction):
        row = self.list.currentRow()
        if direction<0 and row>0:
            self.list.setCurrentRow(row-1)
        elif direction>0 and row<self.list.count()-1:
            self.list.setCurrentRow(row+1)

    def submit(self):
        row = self.list.currentRow()
        if row<0:
            return

        # print(self.filters[self.list.currentItem().text()])

        # create new filter instance (rest is handled by events)
        self.filters[self.list.currentItem().text()]()

        self.close()

