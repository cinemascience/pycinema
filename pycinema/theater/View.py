from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QSpacerItem
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QToolBar 

class View(QtWidgets.QFrame):

    s_close = QtCore.Signal(QtWidgets.QFrame, name='close')
    s_splitH = QtCore.Signal(QtWidgets.QFrame,name='splitH')
    s_splitV = QtCore.Signal(QtWidgets.QFrame,name='splitV')

    def __init__(self, content_layout='V'):
        super().__init__()

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(2,2,2,2)

        # toolbar
        self.toolbar = QToolBar()
        self.toolbar.setStyleSheet("border-style: raised")
        self.toolbar.setIconSize(QtCore.QSize(12,12))
        self.layout().addWidget(self.toolbar)

        # label
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("border-style: ridge;margin:2px 2px")
        font = self.title.font()
        font.setPointSize(10)
        self.title.setFont(font)
        self.toolbar.addWidget(self.title)

        # spacer
        self.spacer = QtWidgets.QLabel()
        self.spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.toolbar.addWidget(self.spacer)

        # buttons
        style = "border-radius: 6;background-color: #808080;border :1px solid #808080;"
        self.button_c = QtWidgets.QToolButton()
        self.button_c.setIcon(QtGui.QIcon("pycinema/theater/icon/cancel.svg"))
        self.button_c.setFixedSize(12,12)
        self.button_c.setStyleSheet(style)
        self.button_c.setToolTip("Close view")
        self.button_c.clicked.connect(self.emitClose)

        self.button_h = QtWidgets.QToolButton()
        self.button_h.setIcon(QtGui.QIcon("pycinema/theater/icon/horizontal-split.png"))
        self.button_h.setFixedSize(12,12)
        self.button_h.setStyleSheet(style)
        self.button_h.setToolTip("Horizontal split")
        self.button_h.clicked.connect(self.emitSplitH)

        self.button_v = QtWidgets.QToolButton()
        self.button_v.setIcon(QtGui.QIcon("pycinema/theater/icon/vertical-split.png"))
        self.button_v.setFixedSize(12,12)
        self.button_v.setStyleSheet(style)
        self.button_v.setToolTip("Vertical split")
        self.button_v.clicked.connect(self.emitSplitV)

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

    # def __del__(self):
    #   print('del VIEW')

    def emitClose(self):
      self.s_close.emit(self)
    def emitSplitH(self):
      self.s_splitH.emit(self)
    def emitSplitV(self):
      self.s_splitV.emit(self)

    def setTitle(self,text):
        self.title.setText(text)
