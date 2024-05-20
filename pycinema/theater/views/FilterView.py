from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.View import View
import logging as log

class FilterView(View):

    def __init__(self, filter):
        super().__init__()
        self.filter = filter
        self.setTitle(filter.id)
        widgets = filter.generateWidgets()
        buttons = []
        if isinstance(widgets,list):
          buttons=widgets[1]
          widgets=widgets[0]
        self.content.layout().addWidget(widgets,1)
        for button in buttons:
          self.toolbar.insertWidget(self.toolbar.actions()[2],button)
        if len(buttons):
          self.toolbar.insertSeparator(self.toolbar.actions()[-3])

        self.filter.on('filter_deleted', self.on_filter_deleted)

    def on_filter_deleted(self,filter):
        if self.filter != filter: return
        self.filter.off('filter_deleted', self.on_filter_deleted)
        self.s_close.emit(self)

    def export(self):
      return self.id + ' = pycinema.theater.views.'+self.__class__.__name__+'( '+self.filter.id+' )\n'
