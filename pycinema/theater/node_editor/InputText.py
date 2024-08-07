from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.node_editor.NodeEditorStyle import NodeEditorStyle as NES
from pycinema.theater.Icons import Icons
from pycinema import Port

class InputText(QtWidgets.QWidget):

    def __init__(self, port, width, parent=None):
        super().__init__(parent)
        self.port = port

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.update_callback = lambda value: self.updateWidget()

        self.setMinimumWidth(width)
        self.setMaximumWidth(width)

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setContentsMargins(0,0,0,0)

        self.label = QtWidgets.QLineEdit()
        self.label.setText(port.name)
        self.label.setReadOnly(True)
        self.label.setStyleSheet('border:0;background:transparent;color:'+NES.COLOR_NORMAL_)

        self.skip = False

        fm = QtGui.QFontMetrics(self.label.font())
        pixelsWide = fm.tightBoundingRect(port.name)

        if port.is_input:
            self.label.setFixedWidth(pixelsWide.width()+10)
            self.layout().addWidget(self.label)

            self.edit = QtWidgets.QLineEdit()
            self.edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.edit.setStyleSheet('border:0;background:transparent;color:'+NES.COLOR_NORMAL_)
            self.edit.setReadOnly(False)

            self.edit.editingFinished.connect(lambda: self.setValue(self.edit.text()))
            self.layout().addWidget(self.edit,1)
            port.on('value_set', self.update_callback)
            self.updateWidget()

            if port.name in ['file','path']:
              def get_path():
                path = QtWidgets.QFileDialog.getExistingDirectory(None,"Select Cinema Database")
                if path: port.set(path)
              select_folder_btn = self.edit.addAction(Icons.toQIcon(Icons.icon_directory), QtWidgets.QLineEdit.TrailingPosition);
              select_folder_btn.triggered.connect(get_path)
              self.edit.children()[-1].setCursor(QtCore.Qt.PointingHandCursor)
        else:
            self.edit = QtWidgets.QLineEdit()
            self.edit.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.edit.setStyleSheet('border:0;background:transparent;color:'+NES.COLOR_DISABLED_)
            self.edit.setReadOnly(True)
            self.layout().addWidget(self.edit,1)
            port.on('value_set', self.update_callback)
            self.updateWidget()

            self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.layout().addWidget(self.label,1)

    def setValue(self,text):
        if self.edit.isReadOnly(): return

        v = None
        try:
            if self.port.type==str:
                v = text
            elif self.port.type==int:
                v = int(text)
            elif self.port.type==float:
                v = float(text)
            else:
                v = eval(text)
        except:
            v = None

        self.skip = True
        self.port.set(v)
        self.skip = False

    def updateWidget(self):
        if self.skip:
            return

        text = self.port.get()
        if text != self.edit.text():
            self.edit.setText(str(text))

        mustBeReadOnly = isinstance(self.port._value, Port) or not self.port.is_input
        isReadOnly = self.edit.isReadOnly()
        if mustBeReadOnly and not isReadOnly:
            self.edit.setStyleSheet('border:0;background:transparent;color:'+NES.COLOR_DISABLED_)
            self.edit.setReadOnly(True)
        elif not mustBeReadOnly and isReadOnly:
            self.edit.setStyleSheet('border:0;background:transparent;color:'+NES.COLOR_NORMAL_)
            self.edit.setReadOnly(False)

    def keyPressEvent(self,event):
      event.accept()
