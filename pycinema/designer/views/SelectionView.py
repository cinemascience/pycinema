from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.designer.View import View

from pycinema.designer.views.NodeView import NodeView
from pycinema.designer.views.TableView import TableView
from pycinema.designer.views.ImageView import ImageView
from pycinema.designer.views.ParameterView import ParameterView
from pycinema.designer.views.ColorMappingView import ColorMappingView
from pycinema.designer.views.ParallelCoordinatesView import ParallelCoordinatesView

class SelectionButton(QtWidgets.QPushButton):
  def __init__(self,name,parent,cls):
    super().__init__(name,parent)
    self.cls = cls
    self.clicked.connect(self.replaceView)

  def replaceView(self):
    self.parent().parent().replaceView(self.parent(),self.cls)

class SelectionView(View):
  def __init__(self):
    super().__init__()
    self.setTitle(self.__class__.__name__)

    self.content.layout().addWidget(QtWidgets.QLabel(),1)

    for cls in [NodeView,TableView,ImageView,ParameterView,ColorMappingView,ParallelCoordinatesView]:
      self.layout().addWidget( SelectionButton(cls.__name__, self, cls) )

    self.layout().addWidget(QtWidgets.QLabel(),1)
