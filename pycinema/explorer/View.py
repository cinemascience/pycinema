from PySide6 import QtCore, QtWidgets, QtGui

class View(QtWidgets.QFrame):

    s_close = QtCore.Signal(QtWidgets.QFrame, name='close')
    s_splitH = QtCore.Signal(name='splitH')
    s_splitV = QtCore.Signal(name='splitV')

    def __init__(self, content_layout='V'):
        super().__init__()

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0,0,0,0)

        self.toolbar = QtWidgets.QFrame()
        self.toolbar.setLayout(QtWidgets.QHBoxLayout())
        # self.toolbar.layout().setContentsMargins(10,2,10,0)
        self.toolbar.layout().setSpacing(0)
        self.toolbar.layout().setContentsMargins(0,0,0,0)

        self.title = QtWidgets.QLabel()
        self.toolbar.setStyleSheet("margin:0px;padding:2px 4px;")
        self.toolbar.layout().addWidget(self.title,1)

        self.button_c = QtWidgets.QPushButton("X")
        self.button_h = QtWidgets.QPushButton("H")
        self.button_v = QtWidgets.QPushButton("V")
        self.button_c.clicked.connect(lambda: self.s_close.emit(self))
        self.button_h.clicked.connect(lambda: self.s_splitH.emit())
        self.button_v.clicked.connect(lambda: self.s_splitV.emit())
        self.toolbar.layout().addWidget(self.button_c)
        self.toolbar.layout().addWidget(self.button_h)
        self.toolbar.layout().addWidget(self.button_v)
        self.layout().addWidget(self.toolbar)

        self.content = QtWidgets.QFrame()
        if content_layout == 'V':
            self.content.setLayout(QtWidgets.QVBoxLayout())
        elif content_layout == 'H':
            self.content.setLayout(QtWidgets.QHBoxLayout())
        elif content_layout == 'G':
            self.content.setLayout(QtWidgets.QGridLayout())
        self.content.layout().setContentsMargins(10,0,10,0)
        self.layout().addWidget(self.content,1)

    def setTitle(self,text):
        self.title.setText(text)
