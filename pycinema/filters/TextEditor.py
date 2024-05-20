from pycinema import Filter

try:
  from PySide6 import QtGui, QtCore, QtWidgets
  from pycinema.theater.Icons import Icons
except Exception:
  pass

class TextEditor(Filter):

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

        button = QtWidgets.QToolButton()
        button.setIcon( Icons.toQIcon(Icons.icon_save) )
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.setFixedSize(18,18)
        button.setToolTip('Save')
        button.clicked.connect( lambda: self.inputs.text.set(widget.toPlainText()) )

        return [widget,[button]]

    def _update(self):
        text = self.inputs.text.get()
        self.outputs.text.set( text )

        for w in self.widgets:
          w.setText(text)

        return 1
