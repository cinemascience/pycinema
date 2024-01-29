from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.View import View

class FilterView(View):

    def __init__(self, filter):
        super().__init__()
        self.filter = filter
        self.setTitle(filter.id)
        widgets = filter.generateWidgets()
        self.content.layout().addWidget(widgets,1)
        self.filter.on('filter_deleted', self.on_filter_deleted)

    def on_filter_deleted(self,filter):
        if self.filter != filter: return
        self.filter.off('filter_deleted', self.on_filter_deleted)
        self.s_close.emit(self)
