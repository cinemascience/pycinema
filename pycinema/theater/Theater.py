from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QMessageBox
import logging as log

import pycinema
from pycinema.theater import View
from pycinema.theater.ViewFrame import *
from pycinema.theater.FilterBrowser import *
from pycinema.theater.views.NodeEditorView import *
from pycinema.theater.Icons import Icons
from pycinema.theater.node_editor.NodeEditorStyle import NodeEditorStyle
from pycinema.workspace.BrowseCinemaDatabase import *
from pycinema.workspace.ExploreCinemaDatabase import *
from pycinema.workspace.ViewCinemaDatabase import *

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
        button_viewCDB.triggered.connect(self.runWorkspaceOnCDB)

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
                if iPort.valueIsPort():
                    script += filter.id + '.inputs.'+iPortName+ '.set(' + iPort._value.parent.id +'.outputs.'+ iPort._value.name +', False)\n'
                elif iPort.valueIsPortList():
                    script += filter.id + '.inputs.'+iPortName+ '.set([' + ','.join([p.parent.id +'.outputs.'+p.name for p in iPort._value]) +'], False)\n'
                else:
                    v = iPort.get()
                    if iPort.type == int or iPort.type == float:
                        script += filter.id + '.inputs.'+iPortName+ '.set(' + str(v) +', False)\n'
                    elif iPort.type == str and len(v.splitlines())<2:
                        script += filter.id + '.inputs.'+iPortName+ '.set("' + str(v) +'", False)\n'
                    elif iPort.type == str and len(v.splitlines())>1:
                        script += filter.id + '.inputs.'+iPortName+ '.set(\'\'\'' + str(v) +'\'\'\', False)\n'
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

    def runWorkspaceOnCDB(self, wfname, path=None):
        if not path:
          path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Cinema Database")
        if not path:
          return

        self.reset(True)

        # default case
        workspace = BrowseCinemaDatabase() 
        if wfname in ['view','explore', 'browse']:

            # if we've identified a valid use case, create the correct instance 
            if wfname == "browse":
                workspace = BrowseCinemaDatabase() 
            elif wfname == "view":
                workspace = ViewCinemaDatabase() 
            else:
                # default
                workspace = ExploreCinemaDatabase() 
        else:
            log.warning("workspace \'" + wfname + "\' not recognized")

        workspace.initializeScript( filename=path )

        self.setWindowTitle("Cinema:" + wfname + " (" + path + ")")
        self.executeScript(workspace.getScript())

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

          elif args[0].endswith('.cdb') or args[0].endswith('.cdb/'):
            Theater.instance.runWorkspaceOnCDB('browse', args[0])

          elif args[0] in ['view','explore', 'browse']:

            path = None
            if len(args)==2 and isinstance(args[1], str): 
                path=args[1]

            if not path: 
                path=QtWidgets.QFileDialog.getExistingDirectory(Theater.instance, "Select Cinema Database")

            if path:
              Theater.instance.runWorkspaceOnCDB(args[0], path)
          else:
            log.warning("workspace \'" + args[0] + "\' not recognized")

        sys.exit(app.exec())
