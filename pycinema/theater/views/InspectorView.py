from pycinema import Filter

from PySide6 import QtCore, QtWidgets, QtGui

from .FilterView import FilterView

import pprint

class InspectorView(Filter, FilterView):

    def __init__(self):
        FilterView.__init__(
          self,
          filter=self,
          delete_filter_on_close = True
        )

        Filter.__init__(
          self,
          inputs={
            'object': None
          }
        )

    def generateWidgets(self):
        self.editor = QtWidgets.QTextEdit()
        self.content.layout().addWidget(self.editor)

    def _update(self):
        x = pprint.pformat(self.inputs.object.get(), indent=2)
        self.editor.setText(x)
        # self.editor.setText(str(self.inputs.object.get()))
        return 1
