from .ViewFrame import *
from .View import *
from .NodeView import *

from .ImageViewer import *
from .ParameterViewer import *

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

        toolbar.addAction(QtGui.QAction("Save", self))
        toolbar.addAction(QtGui.QAction("Filters", self))

        # self.setCentralWidget(ViewFrame(view=NodeView(),root=True))
        self.setCentralWidget(ViewFrame(view=SelectionView(),root=True))

    def openCDB(self, s):
        fileName = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Cinema Database")
        if not fileName:
            return

        frame = self.centralWidget()
        frame.s_splitH(frame.widget(0))
        pv = frame.widget(0).widget(0).convert(ParameterViewer).filter
        iv = frame.widget(1).widget(0).convert(ImageViewer).filter

        x = pycinema.filters.CinemaDatabaseReader()
        x.inputs.path.set(fileName)

        pv.inputs.table.set(x.outputs.table)

        y = pycinema.filters.DatabaseQuery()
        y.inputs.table.set(x.outputs.table)
        y.inputs.sql.set(pv.outputs.sql)

        z = pycinema.filters.ImageReader()
        z.inputs.table.set(y.outputs.table)

        c = pycinema.filters.DepthCompositing()
        c.inputs.images_a.set(z.outputs.images)
        c.inputs.composite_by_meta.set(pv.outputs.composite_by_meta)

        a = pycinema.filters.Annotation()
        a.inputs.images.set(c.outputs.images)

        iv.inputs.images.set(a.outputs.images)

class Explorer():

    def __init__(self):

        # show UI
        app = QtWidgets.QApplication([])

        main_window = _Explorer()
        main_window.resize(1024, 900)
        main_window.show()

        sys.exit(app.exec())
