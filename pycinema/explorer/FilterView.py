from PySide6 import QtCore, QtWidgets, QtGui

from .View import *
from pycinema import Filter

class ViewFilter(Filter):
    def __init__(self, inputs={}, outputs={}):
        super().__init__(inputs=inputs,outputs=outputs)

class FilterView(View):

    def __init__(self, filter=filter):
        super().__init__()

        self.filter = filter(self)
        self.setTitle(self.filter.id)
        self.filter.on('filter_deleted', lambda f: self.closeF(f))

        self.s_close.connect(lambda view: self.closeV())

    def closeF(self,filter):
        if self.filter == filter:
            self.filter = None
            self.s_close.emit(self)

    def closeV(self):
        if self.filter:
            self.filter.delete()
