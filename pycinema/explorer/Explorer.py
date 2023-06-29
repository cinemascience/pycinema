from .ViewFrame import *
from .View import *
from .NodeView import *

from .ImageViewer import *
from .ParameterViewer import *
from .FilterView import ViewFilter

from .FilterBrowser import *

from .Application import *

import pycinema
import pycinema.filters

from PySide6 import QtCore, QtWidgets, QtGui
import sys

class _Explorer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        toolbar = QtWidgets.QToolBar("My main toolbar")
        self.addToolBar(toolbar)

        button_openCDB = QtGui.QAction("Open", self)
        button_openCDB.setStatusTip("open local cinema database")
        button_openCDB.triggered.connect(self.onOpenCDB)
        toolbar.addAction(button_openCDB)

        button_save = QtGui.QAction("Save", self)
        button_save.setStatusTip("save script")
        button_save.triggered.connect(self.saveScript)
        toolbar.addAction(button_save)

        button_load = QtGui.QAction("Load", self)
        button_load.setStatusTip("load script")
        button_load.triggered.connect(self.loadScript)
        toolbar.addAction(button_load)

        button_filters = QtGui.QAction("Filters", self)
        button_filters.setStatusTip("Open Filter Browser")
        button_filters.triggered.connect(self.showFilterBrowser)
        toolbar.addAction(button_filters)

        button_toggle_ne = QtGui.QAction("Toggle Node Editor", self)
        button_toggle_ne.setStatusTip("Show/Hides Node Editor")
        button_toggle_ne.triggered.connect(self.toggleNodeEditor)
        toolbar.addAction(button_toggle_ne)

        self.nodeView = NodeView()
        self.setCentralWidget(ViewFrame(view=self.nodeView,root=True))
        # self.setCentralWidget(ViewFrame(view=SelectionView(),root=True))

    def toggleNodeEditor(self):
        current_state = self.nodeView.isVisible()
        self.nodeView.setVisible(not current_state)

    def showFilterBrowser(self):
        dialog = FilterBrowser()
        dialog.exec()

    def saveScript(self):

        script = '''
import pycinema
import pycinema.filters
import pycinema.explorer

        '''

        script += '\n# pycinema attributes\n'
        script += '\npycinema_version = ' + pycinema.__version__ + '\n'
        script += '\n'
        script += '\n# layout\n'
        script += self.centralWidget().id+' = pycinema.explorer.Explorer.window.centralWidget()\n'
        script += self.centralWidget().export()

        script += '\n# filters\n'
        for _,filter in pycinema.Filter._filters.items():
            if not isinstance(filter,ViewFilter):
                script += filter.id + ' = pycinema.filters.' + filter.__class__.__name__+'()\n'

        script += '\n# properties\n'
        for _,filter in pycinema.Filter._filters.items():
            for iPortName, iPort in filter.inputs.ports():
                if isinstance(iPort._value, pycinema.Port):
                    script += filter.id + '.inputs.'+iPortName+ '.set(' + iPort._value.parent.id +'.outputs.'+ iPort._value.name +', False)\n'
                else:
                    v = iPort.get()
                    if iPort.type == int or iPort.type == float:
                        script += filter.id + '.inputs.'+iPortName+ '.set(' + str(v) +', False)\n'
                    elif iPort.type == str:
                        script += filter.id + '.inputs.'+iPortName+ '.set("' + str(v) +'", False)\n'
                    else:
                        script += filter.id + '.inputs.'+iPortName+ '.set(' + str(v) +', False)\n'

        script += '\n# execute pipeline\n'
        for _,filter in pycinema.Filter._filters.items():
            script += filter.id + '.update()\n'
            break

        script_file_name = QtWidgets.QFileDialog.getSaveFileName(self, "Save Script")
        if len(script_file_name[0])>0:
            try:
                f = open(script_file_name[0], "w")
                f.write(script)
                f.close()
            except:
                return

        return script

    def onOpenCDB(self, s):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Cinema Database")
        if not path:
            return

        self.openCDB(path)

    def openCDB(self, path):
        app = Application('view', filepath=path)
        self.executeScript(app.getScript())

    def executeScript(self, script):
        namespace = {}
        lines = script.splitlines()
        def call(idx):
            if len(lines)<=idx:
                self.nodeView.view.auto_layout = True
                self.nodeView.view.auto_connect = True
                self.nodeView.view.skip_layout_animation = True
                self.nodeView.view.computeLayout()
                self.nodeView.view.skip_layout_animation = False
                self.nodeView.view.fitInView()
                return
            exec(lines[idx], namespace)
            QtCore.QTimer.singleShot(0, lambda: call(idx+1))
        QtCore.QTimer.singleShot(0, lambda: call(0))

    def loadScript(self):
        script_file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Load Script")
        if len(script_file_name[0])>0:
            try:
                script_file = open(script_file_name[0], "r")
                script = script_file.read()
                script_file.close()
                self.nodeView.view.auto_layout = False
                self.nodeView.view.auto_connect = False
                self.executeScript(script)
            except:
                return

class Explorer():

    window = None

    def __init__(self, filepath=None):

        # show UI
        app = QtWidgets.QApplication([])

        Explorer.window = _Explorer()
        Explorer.window.resize(1024, 900)
        Explorer.window.show()

        if filepath:
            Explorer.window.openCDB(filepath)

        sys.exit(app.exec())
