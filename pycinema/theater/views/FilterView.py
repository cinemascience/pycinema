from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.View import View
import logging as log

class FilterView(View):

    def __init__(self, filter, delete_filter_on_close = False):
        super().__init__()
        self.delete_filter_on_close = delete_filter_on_close
        self.filter = filter
        self.filter.on('filter_deleted', self.on_filter_deleted)
        self.filter.on('filter_created', self.on_filter_created)
        self.s_close.connect(self.on_view_close)

        self.generateWidgets()

    def __del__(self):
        log.debug('del FilterView')

    def generateWidgets(self):
        self.frame = QtWidgets.QFrame()
        self.frame.setLayout(QtWidgets.QVBoxLayout())
        self.content.layout().addWidget(self.frame,1)

    def on_filter_created(self,filter):
        if self.filter != filter:
            return
        self.setTitle(filter.id)

    def on_filter_deleted(self,filter):
        if self.filter != filter:
            return
        self.s_close.disconnect(self.on_view_close)
        self.filter.off('filter_deleted', self.on_filter_deleted)
        self.s_close.emit(self)

    def on_view_close(self):
        self.s_close.disconnect(self.on_view_close)
        self.filter.off('filter_deleted', self.on_filter_deleted)
        if self.delete_filter_on_close:
            self.filter.delete()
