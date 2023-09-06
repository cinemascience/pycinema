from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.View import View
from pycinema.theater.ViewStyle import ViewStyle
from pycinema.theater.views import SelectionView
from pycinema.theater.views import FilterView
from pycinema.theater.views import NodeView
from pycinema import Filter

class ViewFrame(QtWidgets.QSplitter):

  id_counter = 0

  def __init__(self, orientation=QtCore.Qt.Horizontal,root=False):
      super().__init__()
      self.root = root
      self.setChildrenCollapsible(False)
      self.setStyleSheet(ViewStyle.get_style_sheet())
      self.id = 'vf'+str(ViewFrame.id_counter)
      ViewFrame.id_counter += 1
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
      frame = ViewFrame()
      self.insertWidget(idx, frame)
      return frame

  def insertView(self,idx,view):
      self.connectView(view)
      self.insertWidget(idx, view)
      return view

  def s_close(self, widget):
    idx = self.indexOf(widget)
    widget.setParent(None)
    # if isinstance(widget, ViewFrame):
    #   widget.deleteLater()
    # else:
    #   self.disconnectView(widget)

    if self.count()<1:
      if self.root:
        self.insertView(0, SelectionView())
      else:
        self.parent().s_close(self)

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
      newFrame = ViewFrame(orientation=orientation)
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

  def replaceView(self,view,cls):
    # print(view,cls)
    idx = self.indexOf(view)
    sizes = self.sizes()

    self.disconnectView(view)
    view.setParent(None)

    newView = cls()
    self.insertView(idx,newView)

    self.setSizes(sizes)

    if issubclass(cls,Filter) and issubclass(cls,FilterView):
      return newView.filter
    else:
      return newView

  def export(self):
    r = ''
    if self.orientation()==QtCore.Qt.Vertical:
      r = self.id + '.setVerticalOrientation()\n'
    else:
      r = self.id + '.setHorizontalOrientation()\n'

    for i in range(0,self.count()):
      w = self.widget(i)
      if isinstance(w,ViewFrame):
        r += w.id + ' = ' + self.id + '.insertFrame('+str(i)+')\n'
        r += w.export()
      elif isinstance(w,Filter):
        r += w.id + ' = ' + self.id+'.insertView( '+str(i)+', pycinema.theater.views.'+w.__class__.__name__+'() )\n'
      else:
        r += self.id+'.insertView( '+str(i)+', pycinema.theater.views.'+w.__class__.__name__+'() )\n'

    r += self.id + '.setSizes('+str(self.sizes())+')\n'
    return r

