from .ViewFrame import *
from .View import *
from .NodeView import *

from .ImageViewer import *
from .ParameterViewer import *
from .FilterView import ViewFilter

from .FilterBrowser import *

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
        button_openCDB.triggered.connect(self.openCDB)
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

    def openCDB(self, s):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Cinema Database")
        if not path:
            return

        script = '''
import pycinema
import pycinema.filters
import pycinema.explorer

# layout
vf0 = pycinema.explorer.Explorer.window.centralWidget()
vf0.s_splitH()
vf1 = vf0.widget(0)
vf2 = vf0.widget(1)
vf2.s_splitH()
vf3 = vf2.widget(0)
vf3.s_splitV()
vf5 = vf3.widget(0)
ParameterViewer_0 = vf5.convert( pycinema.explorer.ParameterViewer )
vf6 = vf3.widget(1)
vf6.s_splitV()
vf7 = vf6.widget(0)
TableViewer_0 = vf7.convert( pycinema.explorer.TableViewer )
vf8 = vf6.widget(1)
ColorMappingViewer_0 = vf8.convert( pycinema.explorer.ColorMappingViewer )
vf4 = vf2.widget(1)
ImageViewer_0 = vf4.convert( pycinema.explorer.ImageViewer )

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()

# properties
ParameterViewer_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableViewer_0.inputs.table.set(TableQuery_0.outputs.table, False)
ColorMappingViewer_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ImageViewer_0.inputs.images.set(ColorMappingViewer_0.outputs.images, False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set(ParameterViewer_0.outputs.sql, False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.composite_by_meta.set(ParameterViewer_0.outputs.composite_by_meta, False)
vf1.widget(0).hide()
'''

        script += 'CinemaDatabaseReader_0.inputs.path.set("'+path+'", False)\n'
        script += 'CinemaDatabaseReader_0.update()'

        self.executeScript(script)

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

    def __init__(self):

        # show UI
        app = QtWidgets.QApplication([])

        Explorer.window = _Explorer()
        Explorer.window.resize(1024, 900)
        Explorer.window.show()

        sys.exit(app.exec())
