from PySide6 import QtCore, QtWidgets, QtGui

from .View import *
# from .ImageView import *
# from .TableView import *
# from .ParameterView import *

from .NodeView import NodeView
from .FilterView import *
from .TableViewer import *
from .ImageViewer import *
from .ParameterViewer import *

import pycinema

def _convert(obj,cls):
  return lambda: obj.convert(cls)

class SelectionView(View):

    def __init__(self):
        super().__init__()
        self.setTitle(self.__class__.__name__)

        self.content.layout().addWidget(QtWidgets.QLabel(),1)

        for cls in [TableViewer,ImageViewer,ParameterViewer]:
            button = QtWidgets.QPushButton(cls.__name__, self)
            button.clicked.connect(_convert(self,cls))
            self.layout().addWidget(button)

        self.layout().addWidget(QtWidgets.QLabel(),1)

    def convert(self,cls):
        viewFrame = self.parent()
        self.setParent(None)
        if issubclass(cls,pycinema.Core.Filter):
            return viewFrame.setView(FilterView(cls))
        else:
            return viewFrame.setView(cls())

