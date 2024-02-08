from PySide6 import QtCore, QtWidgets, QtGui
from pycinema.theater.Icons import Icons

class View(QtWidgets.QFrame):

    s_close = QtCore.Signal(QtWidgets.QFrame, name='close')
    s_splitH = QtCore.Signal(QtWidgets.QFrame,name='splitH')
    s_splitV = QtCore.Signal(QtWidgets.QFrame,name='splitV')

    id_counter = 0

    def __init__(self, content_layout='V'):
        super().__init__()

        self.id = 'view'+str(View.id_counter)
        View.id_counter += 1

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(4,2,4,0)

        # toolbar
        self.toolbar = QtWidgets.QToolBar()
        self.layout().addWidget(self.toolbar)

        # label
        self.title = QtWidgets.QLabel()
        font = self.title.font()
        font.setPointSize(10)
        self.title.setFont(font)
        self.toolbar.addWidget(self.title)

        # spacer
        self.spacer = QtWidgets.QLabel()
        self.spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.toolbar.addWidget(self.spacer)

        def createButton(tooltip,icon,slot):
          button = QtWidgets.QToolButton()
          button.setIcon( Icons.toQIcon(icon) )
          button.setCursor(QtCore.Qt.PointingHandCursor)
          button.setFixedSize(18,18)
          button.setToolTip(tooltip)
          button.clicked.connect(slot)
          return button

        # buttons
        self.button_c = createButton('Close View', Icons.icon_close, self.emitClose)
        self.button_h = createButton('Add View on the Right', Icons.icon_split_h, self.emitSplitH)
        self.button_v = createButton('Add View Below', Icons.icon_split_v, self.emitSplitV)

        self.toolbar.addWidget(self.button_c)
        self.toolbar.addWidget(self.button_h)
        self.toolbar.addWidget(self.button_v)

        self.content = QtWidgets.QFrame()
        if content_layout == 'V':
            self.content.setLayout(QtWidgets.QVBoxLayout())
        elif content_layout == 'H':
            self.content.setLayout(QtWidgets.QHBoxLayout())
        elif content_layout == 'G':
            self.content.setLayout(QtWidgets.QGridLayout())

        self.layout().addWidget(self.content,1)

    def emitClose(self):
      self.s_close.emit(self)
    def emitSplitH(self):
      self.s_splitH.emit(self)
    def emitSplitV(self):
      self.s_splitV.emit(self)

    def setTitle(self,text):
        self.title.setText(text)

    def export(self):
      return self.id + ' = pycinema.theater.views.'+self.__class__.__name__+'()\n'
