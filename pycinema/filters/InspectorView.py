from pycinema import Filter
from PySide6 import QtCore, QtWidgets, QtGui

import pprint

class InspectorView(Filter):
  def __init__(self):
    self.text = ''
    self.widgets = []
    Filter.__init__(
      self,
      inputs={
        'object': None
      }
    )

  def generateWidgets(self):
    widget = QtWidgets.QTextEdit()
    self.widgets.append(widget)
    widget.setText(self.text)
    return widget

  def _update(self):
    self.text = pprint.pformat(self.inputs.object.get(), indent=2)
    for w in self.widgets:
      w.setText(self.text)
    return 1
