from PySide6 import QtCore, QtWidgets, QtGui

from .SelectionView import *
from .FilterView import *
from .NodeView import *
from pycinema import Filter

class ViewFrame(QtWidgets.QSplitter):

    id_counter = 0

    def __init__(self, view, root=False):
        super().__init__()
        self.root = root
        self.setChildrenCollapsible(False)
        self.setStyleSheet("QSplitter::handle {background-color: gray;}")
        self.id = 'vf'+str(ViewFrame.id_counter)
        ViewFrame.id_counter += 1

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

    def split(self,orientation):
        view = self.widget(0)
        view.s_close.disconnect(self.s_close)
        view.s_splitH.disconnect(self.s_splitH)
        view.s_splitV.disconnect(self.s_splitV)

        self.setOrientation(orientation)
        self.insertWidget(0, ViewFrame(view=view))
        self.insertWidget(1, ViewFrame(view=SelectionView()))
        w = self.width()
        self.setSizes([w/2,w/2])

    def s_splitH(self):
        self.split(QtCore.Qt.Horizontal)

    def s_splitV(self):
        self.split(QtCore.Qt.Vertical)

    def convert(self,cls):
        view = self.widget(0)
        view.setParent(None)
        if issubclass(cls,Filter):
            return self.setView(FilterView(cls)).filter
        else:
            return self.setView(cls())

    def export(self):
        if self.count()==1:
            view = self.widget(0)
            if view.__class__ == FilterView:
                filter = view.filter
                return filter.id + ' = ' + self.id+'.convert( pycinema.explorer.'+view.filter.__class__.__name__+' )\n'
            elif view.__class__ == NodeView:
                return ''
            else:
                return self.id+'.convert( pycinema.explorer.'+view.__class__.__name__+' )\n'
        else:
            r = ''
            if self.orientation()==QtCore.Qt.Horizontal:
                r += self.id+'.s_splitH()\n'
            else:
                r += self.id+'.s_splitV()\n'
            vf0 = self.widget(0)
            vf1 = self.widget(1)
            r += vf0.id+' = '+self.id+'.widget(0)\n'
            r += vf0.export()

            r += vf1.id+' = '+self.id+'.widget(1)\n'
            r += vf1.export()

            return r
