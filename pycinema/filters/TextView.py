from pycinema import Filter
from pycinema.theater.Icons import Icons

from PySide6 import QtCore, QtWidgets, QtGui

class TextModel(QtCore.QObject):
# class TextModel(QtWidgets.QWidget):
  s_changed = QtCore.Signal(str, name='s_changed')

  def __init__(self):
    super().__init__()
    self.text = ''

  def set(self,text):
    if self.text==text:
      return

    self.text = text
    self.s_changed.emit(text)

class TextView(Filter):

    def __init__(self):
        self.model = TextModel()

        Filter.__init__(
          self,
          inputs = {
            'text': ''
          },
          outputs = {
            'text': ''
          }
        )

# self.editor.toPlainText()
    def updateInputFromText(self,text):
      self.inputs.text.set(
        text
      )

    def generateWidgets(self):
        widget = QtWidgets.QTextEdit()
        self.model.s_changed.connect(lambda x: widget.setText(x))
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
        self.model.set( text )
        self.outputs.text.set( text )
        return 1
