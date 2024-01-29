from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.View import View
from pycinema.theater.ViewStyle import ViewStyle
from pycinema.theater.Icons import Icons
from pycinema.theater.views import SelectionView
from pycinema.theater.ViewFrame import ViewFrame

from pycinema import Filter

def createButton(parent,tooltip,icon):
  button = QtWidgets.QToolButton(parent)
  button.setIcon( Icons.toQIcon(icon) )
  button.setCursor(QtCore.Qt.PointingHandCursor)
  button.setFixedSize(18,18)
  button.setToolTip(tooltip)
  return button

class TabFrame(QtWidgets.QTabWidget):

  id_counter = 0

  def __init__(self, root=False):
      super().__init__()
      self.root = root
      # self.setStyleSheet(ViewStyle.get_style_sheet())
      self.id = 'tf'+str(TabFrame.id_counter)
      TabFrame.id_counter += 1

      super().insertTab(0,QtWidgets.QWidget(),'  +  ')
      self.currentChanged.connect(self.skipAddTab)

      self.old_idx = -1

      self.tabBar().tabBarClicked.connect(self.tabBarC)
      self.tabBar().tabBarDoubleClicked.connect(self.tabBarDBC)

  def skipAddTab(self,idx):
    if idx==self.count()-1 and self.old_idx>=0 and self.old_idx!=idx:
      temp = self.old_idx
      self.old_idx= idx
      self.setCurrentIndex(temp)

    self.old_idx = idx

  def tabBarC(self,idx):
    if idx!=self.count()-1: return
    vf = ViewFrame(root=True)
    vf.insertView(0,SelectionView())
    self.insertTab(idx,vf)
    self.setCurrentIndex(idx)

  def tabBarDBC(self,idx):
    if idx==self.count()-1: return

    bar = self.tabBar()
    rect = bar.tabRect(idx)
    top_margin = 3
    left_margin = 6
    bar.__edit = QtWidgets.QLineEdit(bar)
    bar.__edit.idx = idx
    bar.__edit.show()
    bar.__edit.move(rect.left() + left_margin, rect.top() + top_margin)
    bar.__edit.resize(rect.width() - 2 * left_margin, rect.height() - 2 * top_margin)
    bar.__edit.setText(bar.tabText(idx))
    bar.__edit.selectAll()
    bar.__edit.setFocus()
    bar.__edit.editingFinished.connect(self.fRenameTab)

  def fRenameTab(self):
    bar = self.tabBar()
    bar.setTabText(bar.__edit.idx, bar.__edit.text())
    bar.__edit.deleteLater()
    bar.__edit = None

  def insertTab(self,idx,widget=None,name=None):
      if widget==None:
        widget = ViewFrame(root=True)
        widget.insertView(0,SelectionView())
      if name==None:
        name = "Layout %d"%self.count()
      super().insertTab(idx,widget,name)
      self.tabBar().setTabButton(
        idx,
        QtWidgets.QTabBar.ButtonPosition.RightSide,
        createButton(self.tabBar(),'Close Tab', Icons.icon_close)
      )
      self.setCurrentIndex(idx)
      return self.widget(idx)

  def export(self):
    r = ''
    for i in range(0,self.count()):
      w = self.widget(i)
      if isinstance(w,ViewFrame):
        r += w.id + ' = ' + self.id + '.insertTab('+str(i)+')\n'
        r += w.export()

    return r

