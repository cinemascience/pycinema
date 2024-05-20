from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.View import View

from pycinema import Filter, filters
from pycinema.theater import views
from pycinema.theater.views.FilterView import FilterView
from pycinema.theater.views.NodeEditorView import NodeEditorView, QtNodeEditorView
import pycinema.theater.TabFrame
import re

class SelectionButton(QtWidgets.QPushButton):
  def __init__(self,name,parent,cls):
    super().__init__(name,parent)
    self.cls = cls
    self.clicked.connect(self.replaceView)

  def replaceView(self):
    self.parent().parent().replaceView(
      self.parent(),
      self.cls()
    )

class ActiveFilterButton(QtWidgets.QPushButton):
  def __init__(self,parent):
    super().__init__('Active Filter',parent)
    self.clicked.connect(self.replaceView)

  def replaceView(self):
    items = QtNodeEditorView.scene.selectedItems()
    if len(items)<1: return
    self.parent().parent().replaceView(self.parent(),items[0].filter)

class AddTabFrameButton(QtWidgets.QPushButton):
  def __init__(self,parent):
    super().__init__('Tab Frame',parent)
    self.clicked.connect(self.convert)

  def convert(self):
    tf = pycinema.theater.TabFrame()
    tf.insertTab(0)
    self.parent().parent().replaceView(self.parent(),tf)

class SelectionView(View):
  def __init__(self):
    super().__init__()
    self.setTitle(self.__class__.__name__)

    self.content.layout().addWidget(QtWidgets.QLabel(),1)

    view_list = [cls for name, cls in filters.__dict__.items() if isinstance(cls,type) and issubclass(cls,Filter) and hasattr(cls,'generateWidgets')]
    view_list.sort(key=lambda x: x.__name__)
    # view_list.insert(0,NodeEditorView)

    l = self.layout()
    l.addWidget( SelectionButton('Pipeline Editor', self, NodeEditorView) )
    l.addWidget( ActiveFilterButton(self) )
    l.addWidget( AddTabFrameButton(self) )

    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    # line.setContentsMargins(0, 20, 0, 20)
    line.setMinimumHeight(20)
    l.addWidget(line)

    for cls in view_list:
      l.addWidget( SelectionButton(re.sub(r'(\w)([A-Z])', r'\1 \2', cls.__name__.replace('View','')), self, cls) )
    l.addWidget(QtWidgets.QLabel(),1)
