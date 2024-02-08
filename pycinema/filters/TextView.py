from pycinema import Filter
from pycinema.theater.Icons import Icons

from PySide6 import QtCore, QtWidgets, QtGui

class TextView(Filter):

    def __init__(self):
        self.widgets = []

        Filter.__init__(
          self,
          inputs = {
            'text': ''
          },
          outputs = {
            'text': ''
          }
        )

    def updateInputFromText(self,text):
      self.inputs.text.set(
        text
      )

    def generateWidgets(self):
        widget = QtWidgets.QTextEdit()
        self.widgets.append(widget)
        return widget

        # self.editor_toolbar = QtWidgets.QToolBar()

        # def createButton(tooltip,icon,slot):
        #   button = QtWidgets.QToolButton()
        #   button.setIcon( Icons.toQIcon(icon) )
        #   button.setCursor(QtCore.Qt.PointingHandCursor)
        #   button.setFixedSize(18,18)
        #   button.setToolTip(tooltip)
        #   button.clicked.connect(slot)
        #   return button

        # self.button_c = createButton('Save', Icons.icon_save, lambda: self.updateInputFromText())

        # self.editor_toolbar.addWidget(self.button_c)
        # self.content.layout().addWidget(self.editor_toolbar)

    def _update(self):
        text = self.inputs.text.get()
        self.outputs.text.set( text )

        for w in self.widgets:
          w.setText(text)

        return 1
