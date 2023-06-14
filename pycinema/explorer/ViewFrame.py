from PySide6 import QtCore, QtWidgets, QtGui

from .SelectionView import *

class ViewFrame(QtWidgets.QSplitter):

    def __init__(self, view, root=False):
        super().__init__()
        self.root = root
        self.setChildrenCollapsible(False)
        self.setStyleSheet("QSplitter::handle {background-color: gray;}")

        self.setView(view)

    def setView(self,view):
        view.s_close.connect(self.s_close)
        view.s_splitH.connect(self.s_splitH)
        view.s_splitV.connect(self.s_splitV)
        self.insertWidget(0, view)
        return view

    def s_close(self, view):
        # in case of root frame replace view with selectionView
        if self.root:
            return

        # find view_ that must be preserved
        view_ = None
        parent = self.parent()
        for i in range(0,2):
            v = parent.widget(i).widget(0)
            if v!=view:
                view_ = v

        # delete view and frame
        nestedViewFrame = view.parent()
        view.setParent(None)
        nestedViewFrame.setParent(None)

        # delete frame and
        nestedViewFrame = view_.parent()
        parent.setView(view_)
        nestedViewFrame.setParent(None)

        # if self.root and self.count()==1:
        #     return

        # view.setParent(None)
        # if self.count()==0:
        #     self.setParent(None)

    def split(self,view,orientation):
        view.s_close.disconnect(self.s_close)
        view.s_splitH.disconnect(self.s_splitH)
        view.s_splitV.disconnect(self.s_splitV)

        self.setOrientation(orientation)
        self.insertWidget(0, ViewFrame(view=view))
        self.insertWidget(1, ViewFrame(view=SelectionView()))

    def s_splitH(self, view):
        self.split(view,QtCore.Qt.Horizontal)

    def s_splitV(self, view):
        self.split(view,QtCore.Qt.Vertical)

    def test(self):
        print('test')
