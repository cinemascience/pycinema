from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QMessageBox

import traceback
import pycinema
from pycinema.theater import View
from pycinema.theater.SplitFrame import *
from pycinema.theater.TabFrame import *
from pycinema.theater.FilterBrowser import *
from pycinema.theater.views.NodeEditorView import *
from pycinema.theater.Icons import Icons
from pycinema.theater.node_editor.NodeEditorStyle import NodeEditorStyle

import sys, os

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
        button_viewCDB.triggered.connect(self.viewDatabase)

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

        self.reset(False)

    def viewDatabase(self,path):
      if not path:
        path = QtWidgets.QFileDialog.getExistingDirectory(Theater.instance, "Select Cinema Database")
      if path:
        Theater.instance.loadScript('./pycinema/scripts/view.py', None, [path])

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

        script += '\n# filters\n'
        for _,filter in pycinema.Filter._filters.items():
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

        script += '\n# layout\n'
        script += self.centralWidget().export()
        script += 'pycinema.theater.Theater.instance.setCentralWidget('+self.centralWidget().id+')\n'

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

    def reset(self, no_views=True):
      Filter._processing = True
      QtNodeEditorView.auto_layout = False
      QtNodeEditorView.auto_connect = False
      for f in list(Filter._filters):
        f.delete()
      Filter._processing = False
      QtNodeEditorView.auto_layout = True
      QtNodeEditorView.auto_connect = True

      if no_views:
        self.centralWidget().setParent(None)
      else:
        vf = SplitFrame()
        vf.insertView(0,NodeEditorView())
        tf = TabFrame()
        tf.insertTab(0,vf)
        self.setCentralWidget(tf)

    def quit(self, no_views=False):
        QtWidgets.QApplication.quit()
        return

    def about(self, no_views=False):
        msgBox = QtWidgets.QMessageBox.about(self, "About", "pycinema v" + pycinema.__version__);
        return

    def executeScript(self, script, args=[]):
        QtNodeEditorView.auto_layout = False
        QtNodeEditorView.auto_connect = False

        variables = {}
        for i,arg in enumerate(args):
          variables['PYCINEMA_ARG_'+str(i)] = arg

        try:
          exec(script,variables)
        except Exception as err:
          traceback.print_exc()
          self.reset(False)

        def call():
          QtNodeEditorView.auto_layout = True
          QtNodeEditorView.auto_connect = True
          QtNodeEditorView.skip_layout_animation = True
          QtNodeEditorView.computeLayout()
          QtNodeEditorView.skip_layout_animation = False
          if not self.centralWidget() or self.centralWidget().count()<1:
            self.centralWidget().insertView(0,NodeEditorView())

          editors = self.centralWidget().findChildren(QtNodeEditorView)
          for editor in editors:
            editor.fitInView()
          return
        QtCore.QTimer.singleShot(0, lambda: call())

    def loadScript(self, script_file_name=None, scriptkey=None, args=[]):
        if not script_file_name:
            script_file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Load Script")[0]
        if script_file_name and len(script_file_name)>0:
            try:
                script_file = open(script_file_name, "r")
                script = script_file.read()
                script_file.close()
                self.reset(True)
                if not scriptkey:
                    self.setWindowTitle("Cinema:Theater (" + script_file_name + ")")
                else:
                    self.setWindowTitle("Cinema:" + scriptkey)
                self.executeScript(script,args)
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

        scriptkey = None 
        if len(args)>0 and isinstance(args[0], str):
          if args[0].endswith('.py'):
            Theater.instance.loadScript(args[0], scriptkey, args[1:])

          elif args[0].endswith('.cdb') or args[0].endswith('.cdb/'):
            script = pycinema.getPathForScript('browse')
            Theater.instance.loadScript(script, scriptkey, [args[0]])

          else: 
            script = pycinema.getPathForScript(args[0])
            if script:
              print("loading script: " + script)
              if len(args)<2:
                args.append(QtWidgets.QFileDialog.getExistingDirectory(Theater.instance, "Select Cinema Database"))

              Theater.instance.loadScript(script, args[0], args[1:])
            else:
              print("no script found for key: '" + args[0] + "'")

        sys.exit(app.exec())
