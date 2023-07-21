from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.designer.View import View
# from .ImageView import *
# from .TableView import *
# from .ParameterView import *

# from .NodeView import NodeView
from pycinema.designer.views.TableView import TableView
from pycinema.designer.views.ImageView import ImageView
from pycinema.designer.views.ParameterView import ParameterView
from pycinema.designer.views.ColorMappingView import ColorMappingView
# from .ImageViewer import *
# from .ColorMappingViewer import *
# from .OpenGLViewer import *

def replaceView(view,cls):
  return lambda: view.parent().replaceView(view,cls)

class SelectionView(View):

    def __init__(self):
        super().__init__()
        self.setTitle(self.__class__.__name__)

        self.content.layout().addWidget(QtWidgets.QLabel(),1)

        for cls in [TableView,ImageView,ParameterView,ColorMappingView]:
            button = QtWidgets.QPushButton(cls.__name__, self)
            button.clicked.connect(replaceView(self,cls))
            self.layout().addWidget(button)

        self.layout().addWidget(QtWidgets.QLabel(),1)
