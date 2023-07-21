from PySide6 import QtCore, QtWidgets, QtGui

import pycinema
from pycinema.designer import View
from pycinema.designer.ViewFrame import *
from pycinema.designer.FilterBrowser import *
from pycinema.designer.node_editor.NodeView import *

import sys

class _Designer(QtWidgets.QMainWindow):
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
        vf = ViewFrame(root=True)
        vf.insertView(0,self.nodeView)
        self.setCentralWidget(vf)

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
import pycinema.designer
import pycinema.designer.views
'''

        script += '\n# layout\n'
        if self.nodeView.isVisible():
          script += 'pycinema.designer.Designer.instance.nodeView.setVisible(True)\n'
        else:
          script += 'pycinema.designer.Designer.instance.nodeView.setVisible(False)\n'
        script += self.centralWidget().id+' = pycinema.designer.Designer.instance.centralWidget()\n'
        script += self.centralWidget().export()

        script += '\n# filters\n'
        for _,filter in pycinema.Filter._filters.items():
            if not isinstance(filter,View):
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

    def reset(self):
      Filter._processing = True
      self.nodeView.view.auto_layout = False
      self.nodeView.view.auto_connect = False
      for f in list(Filter._filters):
        f.delete()
      Filter._processing = False
      self.nodeView.view.auto_layout = True
      self.nodeView.view.auto_connect = True

    def openCDB(self, s):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Cinema Database")
        if not path:
            return

        self.reset()

        script = '''
import pycinema
import pycinema.filters
import pycinema.designer
import pycinema.designer.views

# layout
pycinema.designer.Designer.instance.nodeView.setVisible(False)
vf0 = pycinema.designer.Designer.instance.centralWidget()
vf0.setHorizontalOrientation()
vf1 = vf0.insertFrame(1)
vf1.setVerticalOrientation()
ParameterView_0 = vf1.insertView( 0, pycinema.designer.views.ParameterView() )
TableView_0 = vf1.insertView( 1, pycinema.designer.views.TableView() )
ColorMappingView_0 = vf1.insertView( 2, pycinema.designer.views.ColorMappingView() )
ImageView_0 = vf0.insertView( 2, pycinema.designer.views.ImageView() )
vf0.setSizes([0, 1500, 3000])
vf1.setSizes([1000,1000])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()

# properties
ParameterView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableView_0.inputs.table.set(TableQuery_0.outputs.table, False)
ColorMappingView_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ImageView_0.inputs.images.set(ColorMappingView_0.outputs.images, False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set(ParameterView_0.outputs.sql, False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.composite_by_meta.set(ParameterView_0.outputs.compose, False)
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
                self.reset()
                self.executeScript(script)
            except:
                return

class Designer():

    instance = None

    def __init__(self):

        # show UI
        app = QtWidgets.QApplication([])

        Designer.instance = _Designer()
        Designer.instance.resize(1024, 900)
        Designer.instance.show()

        sys.exit(app.exec())
