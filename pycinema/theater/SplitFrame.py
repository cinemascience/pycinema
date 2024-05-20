from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.View import View
from pycinema.theater.ViewStyle import ViewStyle
from pycinema.theater.views import SelectionView, FilterView
import pycinema.theater.TabFrame
from pycinema import Filter

class SplitFrame(QtWidgets.QSplitter):

  id_counter = 0

  def __init__(self, orientation=QtCore.Qt.Horizontal):
      super().__init__()
      self.setChildrenCollapsible(False)
      self.setStyleSheet(ViewStyle.get_style_sheet())
      self.id = 'splitFrame'+str(SplitFrame.id_counter)
      SplitFrame.id_counter += 1
      self.setOrientation(orientation)

  def setVerticalOrientation(self):
      self.setOrientation(QtCore.Qt.Vertical)

  def setHorizontalOrientation(self):
      self.setOrientation(QtCore.Qt.Horizontal)

  def connectView(self,view):
      view.s_close.connect(self.s_close)
      view.s_splitH.connect(self.s_splitH)
      view.s_splitV.connect(self.s_splitV)

  def disconnectView(self,view):
      view.s_close.disconnect(self.s_close)
      view.s_splitH.disconnect(self.s_splitH)
      view.s_splitV.disconnect(self.s_splitV)

  def insertFrame(self,idx):
      frame = SplitFrame()
      self.insertWidget(idx, frame)
      return frame

  def insertView(self,idx,view):
      if isinstance(view,Filter):
        view = FilterView(view)

      if isinstance(view,View):
        self.connectView(view)

      self.insertWidget(idx, view)
      return view

  def s_close(self, widget):
    idx = self.indexOf(widget)
    widget.setParent(None)

    if self.count()<1:
      if isinstance(self.parent(),SplitFrame):
        self.parent().s_close(self)
      else:
        self.insertView(0, SelectionView())

  def split(self,view,orientation):
    idx = self.indexOf(view)

    width = view.width()
    height = view.height()

    if self.count()<2:
      self.setOrientation(orientation)

    widths = []
    heights = []
    if self.orientation() == orientation:
      for i in range(0,self.count()):
        if i==idx:
          widths.append(width/2)
          widths.append(width/2)
          heights.append(height/2)
          heights.append(height/2)
        else:
          v = self.widget(i)
          widths.append(v.width())
          heights.append(v.height())

      self.insertView(idx+1, SelectionView())

    else:
      for i in range(0,self.count()):
        v = self.widget(i)
        widths.append(v.width())
        heights.append(v.height())

      self.disconnectView(view)
      newFrame = SplitFrame(orientation=orientation)
      newFrame.insertView(0,view)
      newFrame.insertView(1,SelectionView())
      if orientation==QtCore.Qt.Horizontal:
        newFrame.setSizes([width/2,width/2])
      else:
        newFrame.setSizes([height/2,height/2])

      self.insertWidget(idx, newFrame)

    if orientation==QtCore.Qt.Horizontal:
      self.setSizes(widths)
    else:
      self.setSizes(heights)

  def s_splitH(self,view):
    self.split(view,QtCore.Qt.Horizontal)

  def s_splitV(self,view):
    self.split(view,QtCore.Qt.Vertical)

  def replaceView(self,oldView,newView):
    idx = self.indexOf(oldView)
    sizes = self.sizes()
    self.disconnectView(oldView)
    oldView.setParent(None)
    self.insertView(idx,newView)
    self.setSizes(sizes)

  def export(self):
    r = self.id + ' = pycinema.theater.SplitFrame()\n'
    if self.orientation()==QtCore.Qt.Vertical:
      r += self.id + '.setVerticalOrientation()\n'
    else:
      r += self.id + '.setHorizontalOrientation()\n'

    for i in range(0,self.count()):
      w = self.widget(i)
      r += w.export()
      r += self.id+'.insertView( '+str(i)+', '+w.id+' )\n'

    r += self.id + '.setSizes('+str(self.sizes())+')\n'
    return r

