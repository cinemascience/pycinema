from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QMessageBox

import pycinema
from pycinema.theater import View
from pycinema.theater.ViewFrame import *
from pycinema.theater.FilterBrowser import *
from pycinema.theater.views.NodeEditorView import *
from pycinema.theater.Icons import Icons
from pycinema.theater.node_editor.NodeEditorStyle import NodeEditorStyle

import sys

class _Theater(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # init theme
        Icons.update_theme();
        NodeEditorStyle.update_theme();
        ViewStyle.update_theme();

        # make actions
        button_viewCDB = QtGui.QAction("Open Cinema database ...", self)
        button_viewCDB.setStatusTip("open local cinema database")
        button_viewCDB.triggered.connect(self.viewCDB)

        button_save = QtGui.QAction("Save script ...", self)
        button_save.setStatusTip("save script")
        button_save.triggered.connect(self.saveScript)

        button_load = QtGui.QAction("Load script ...", self)
        button_load.setStatusTip("load script")
        button_load.triggered.connect(self.loadScript)

        button_filters = QtGui.QAction("Add filter ...", self)
        button_filters.setStatusTip("Open Filter Browser")
        button_filters.triggered.connect(self.showFilterBrowser)

        button_reset = QtGui.QAction("Reset", self)
        button_reset.setStatusTip("Reset Theater")
        button_reset.triggered.connect(self.reset)

        button_about = QtGui.QAction("About ...", self)
        button_about.setStatusTip("About Cinema")
        button_about.triggered.connect(self.about)

        button_quit = QtGui.QAction("Quit Cinema", self)
        button_quit.setStatusTip("Quit Cinema")
        button_quit.triggered.connect(self.quit)

        # menu
        menuBar = self.menuBar();
        menuBar.setNativeMenuBar(False)
        cinemaMenu = menuBar.addMenu("Cinema")
        cinemaMenu.addAction(button_about)
        cinemaMenu.addAction(button_quit)
        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction(button_viewCDB)
        fileMenu.addAction(button_load)
        fileMenu.addAction(button_save)
        fileMenu.addSeparator()
        fileMenu.addAction(button_reset)
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addAction(button_filters)

        # toolbar = QtWidgets.QToolBar("My main toolbar")
        # self.addToolBar(toolbar)
        # toolbar.addAction(button_viewCDB)
        # toolbar.addAction(button_save)
        # toolbar.addAction(button_load)
        # toolbar.addAction(button_filters)
        # toolbar.addAction(button_reset)
        # toolbar.addAction(button_about)

        # status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        vf = ViewFrame(root=True)
        vf.insertView(0,NodeEditorView())
        self.setCentralWidget(vf)

    def showFilterBrowser(self):
        dialog = FilterBrowser()
        dialog.exec()

    def saveScript(self):

        script = '''import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

'''

        script += '# pycinema settings\n'
        script += 'PYCINEMA = { \'VERSION\' : \'' + pycinema.__version__ + '\'}\n'
        script += '\n# layout\n'
        script += self.centralWidget().id+' = pycinema.theater.Theater.instance.centralWidget()\n'
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
            script_file_name = script_file_name[0]
            if not script_file_name.endswith('.py'):
                script_file_name += '.py'
            try:
                f = open(script_file_name, "w")
                f.write(script)
                f.close()
            except:
                return

        return script

    def reset(self, no_views=False):
      Filter._processing = True
      QtNodeEditorView.auto_layout = False
      QtNodeEditorView.auto_connect = False
      for f in list(Filter._filters):
        f.delete()
      Filter._processing = False
      QtNodeEditorView.auto_layout = True
      QtNodeEditorView.auto_connect = True

      root = self.centralWidget()

      def findAllViews(views,vf):
        for i in range(0,vf.count()):
          w = vf.widget(i)
          if isinstance(w, ViewFrame):
            findAllViews(views,w)
          else:
            views.append(w)

      views = []
      findAllViews(views,root)
      for v in views:
        v.parent().s_close(v)

      if no_views:
        self.centralWidget().widget(0).setParent(None)

    def viewCDB(self, path=None):
        if not path:
          path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Cinema Database")
        if not path:
          return

        self.reset(True)

        script = '''
import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf1 = vf0.insertFrame(0)
vf1.setVerticalOrientation()
ParameterView_0 = vf1.insertView( 0, pycinema.theater.views.ParameterView() )
TableView_0 = vf1.insertView( 1, pycinema.theater.views.TableView() )
ColorMappingView_0 = vf1.insertView( 2, pycinema.theater.views.ColorMappingView() )
ImageView_0 = vf0.insertView( 1, pycinema.theater.views.ImageView() )
vf0.setSizes([300,600])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()
ShaderSSAO_0 = pycinema.filters.ShaderSSAO()
ImageAnnotation_0 = pycinema.filters.ImageAnnotation()

# properties
ParameterView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)

TableView_0.inputs.table.set(ParameterView_0.outputs.table, False)
ImageReader_0.inputs.table.set(ParameterView_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.compose.set(ParameterView_0.outputs.compose, False)
ColorMappingView_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ShaderSSAO_0.inputs.images.set(ColorMappingView_0.outputs.images,False)
ShaderSSAO_0.inputs.samples.set(128,False)
ImageAnnotation_0.inputs.images.set(ShaderSSAO_0.outputs.images, False)
ImageView_0.inputs.images.set(ImageAnnotation_0.outputs.images, False)
'''
        script += 'CinemaDatabaseReader_0.inputs.path.set("'+path+'", False)\n'
        script += 'CinemaDatabaseReader_0.update()'
        self.setWindowTitle("Cinema:View (" + path + ")")
        self.executeScript(script)

    def exploreCDB(self, path=None):
        if not path:
          path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Cinema Database")
        if not path:
          return

        self.reset(True)

        script = '''
import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf1 = vf0.insertFrame(0)
vf1.setVerticalOrientation()
ParallelCoordinatesView_0 = vf1.insertView( 0, pycinema.theater.views.ParallelCoordinatesView() )
TableView_0 = vf1.insertView( 1, pycinema.theater.views.TableView() )
ColorMappingView_0 = vf1.insertView( 2, pycinema.theater.views.ColorMappingView() )
vf2 = vf0.insertFrame(1)
vf2.setVerticalOrientation()
ImageView_0 = vf2.insertView( 0, pycinema.theater.views.ImageView() )
vf0.setSizes([400,600])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()
ShaderSSAO_0 = pycinema.filters.ShaderSSAO()
ImageAnnotation_0 = pycinema.filters.ImageAnnotation()

# properties
ParallelCoordinatesView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableView_0.inputs.table.set(ParallelCoordinatesView_0.outputs.table, False)
ImageView_0.inputs.images.set(ColorMappingView_0.outputs.images, False)
ImageReader_0.inputs.table.set(ParallelCoordinatesView_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.compose.set(ParallelCoordinatesView_0.outputs.compose, False)
ColorMappingView_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ShaderSSAO_0.inputs.images.set(ColorMappingView_0.outputs.images,False)
ShaderSSAO_0.inputs.samples.set(128,False)
ImageAnnotation_0.inputs.images.set(ShaderSSAO_0.outputs.images, False)
ImageView_0.inputs.images.set(ImageAnnotation_0.outputs.images, False)
'''
        script += 'CinemaDatabaseReader_0.inputs.path.set("'+path+'", False)\n'
        script += 'CinemaDatabaseReader_0.update()'
        self.setWindowTitle("Cinema:Explorer (" + path + ")")
        self.executeScript(script)

    def quit(self, no_views=False):
        QtWidgets.QApplication.quit()
        return

    def about(self, no_views=False):
        msgBox = QtWidgets.QMessageBox.about(self, "About", "pycinema v" + pycinema.__version__);
        return

    def executeScript(self, script):
        QtNodeEditorView.auto_layout = False
        QtNodeEditorView.auto_connect = False

        try:
          exec(script)
        except:
          return

        def call():
          QtNodeEditorView.auto_layout = True
          QtNodeEditorView.auto_connect = True
          QtNodeEditorView.skip_layout_animation = True
          QtNodeEditorView.computeLayout()
          QtNodeEditorView.skip_layout_animation = False
          if self.centralWidget().count()<1:
            self.centralWidget().insertView(0,NodeEditorView())

          editors = self.centralWidget().findChildren(QtNodeEditorView)
          for editor in editors:
            editor.fitInView()
          return
        QtCore.QTimer.singleShot(0, lambda: call())

    def loadScript(self, script_file_name=None):
        if not script_file_name:
            script_file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Load Script")[0]
        if script_file_name and len(script_file_name)>0:
            try:
                script_file = open(script_file_name, "r")
                script = script_file.read()
                script_file.close()
                self.reset(True)
                self.setWindowTitle("Cinema:Theater (" + script_file_name + ")")
                self.executeScript(script)
            except:
                return

class Theater():

    instance = None

    def __init__(self, args=[]):

        # show UI
        app = QtWidgets.QApplication([])

        Theater.instance = _Theater()
        Theater.instance.resize(1024, 900)
        Theater.instance.show()

        if len(args)>0 and isinstance(args[0], str):
          if args[0].endswith('.py'):
            Theater.instance.loadScript(args[0])
          elif args[0] in ['view','explorer']:
            path = None
            if len(args)==2 and isinstance(args[1], str): path=args[1]
            if not path: path=QtWidgets.QFileDialog.getExistingDirectory(Theater.instance, "Select Cinema Database")
            if path:
              if args[0]=='view':
                Theater.instance.viewCDB(path)
              else:
                Theater.instance.exploreCDB(path)

        sys.exit(app.exec())
