from pycinema import Filter
from pycinema.theater.Icons import Icons

from PySide6 import QtCore, QtWidgets, QtGui

from .FilterView import FilterView

class TextView(Filter, FilterView):

    def __init__(self):
        FilterView.__init__(
          self,
          filter=self,
          delete_filter_on_close = True
        )

        Filter.__init__(
          self,
          inputs = {
            'text': ''
          },
          outputs = {
            'text': ''
          }
        )

    def updateInputFromText(self):
      self.inputs.text.set(
        self.editor.toPlainText()
      )

    def generateWidgets(self):
        self.editor = QtWidgets.QTextEdit()

        self.editor_toolbar = QtWidgets.QToolBar()

        def createButton(tooltip,icon,slot):
          button = QtWidgets.QToolButton()
          button.setIcon( Icons.toQIcon(icon) )
          button.setCursor(QtCore.Qt.PointingHandCursor)
          button.setFixedSize(18,18)
          button.setToolTip(tooltip)
          button.clicked.connect(slot)
          return button

        self.button_c = createButton('Save', Icons.icon_save, lambda: self.updateInputFromText())

        self.editor_toolbar.addWidget(self.button_c)

        self.content.layout().addWidget(self.editor_toolbar)
        self.content.layout().addWidget(self.editor)

    def _update(self):

        self.editor.setText(
          self.inputs.text.get()
        )

        self.outputs.text.set(
          self.inputs.text.get()
        )

        return 1
