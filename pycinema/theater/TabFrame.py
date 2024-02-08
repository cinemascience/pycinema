from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.View import View
from pycinema.theater.ViewStyle import ViewStyle
from pycinema.theater.Icons import Icons
import pycinema.theater.views.SelectionView
import pycinema.theater.SplitFrame

from pycinema import Filter

def createButton(parent,tooltip,icon):
  button = QtWidgets.QPushButton(parent)
  button.setIcon( Icons.toQIcon(icon) )
  button.setCursor(QtCore.Qt.PointingHandCursor)
  button.setFixedSize(18,18)
  button.setToolTip(tooltip)
  return button

class TabFrame(QtWidgets.QTabWidget):

  id_counter = 0

  def __init__(self):
      super().__init__()
      # self.setStyleSheet(ViewStyle.get_style_sheet())
      self.id = 'tabFrame'+str(TabFrame.id_counter)
      TabFrame.id_counter += 1
      self.tab_counter = 1

      super().insertTab(0,QtWidgets.QWidget(),'  +  ')
      self.currentChanged.connect(self.skipAddTab)

      self.old_idx = -1

      self.tabBar().tabBarClicked.connect(self.tabBarC)
      self.tabBar().tabBarDoubleClicked.connect(self.tabBarDBC)

  def tabRemoved(self,idx):
    if self.count()<2:
      if isinstance(self.parent(),pycinema.theater.SplitFrame):
        self.parent().s_close(self)
      else:
        self.insertTab(0)

    return super().tabRemoved(idx)

  def skipAddTab(self,idx):
    if idx==self.count()-1 and self.old_idx>=0 and self.old_idx!=idx:
      temp = self.old_idx
      self.old_idx= idx
      self.setCurrentIndex(temp)

    self.old_idx = idx

  def tabBarC(self,idx):
    if idx!=self.count()-1: return
    self.insertTab(idx)

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

  def insertTab(self,idx,splitFrame=None,name=None):
      if splitFrame==None:
        splitFrame = pycinema.theater.SplitFrame()
        splitFrame.insertView(0,pycinema.theater.views.SelectionView())

      if name==None:
        name = "Layout %d"%self.tab_counter
        self.tab_counter += 1

      super().insertTab(idx,splitFrame,name)

      closeButton = createButton(self.tabBar(),'Close Tab', Icons.icon_close)
      self.tabBar().setTabButton(
        idx,
        QtWidgets.QTabBar.ButtonPosition.RightSide,
        closeButton
      )
      self.setCurrentIndex(idx)
      closeButton.clicked.connect(lambda: self.removeTab(self.indexOf(splitFrame)))
      return splitFrame

  def export(self):
    r = self.id + ' = pycinema.theater.TabFrame()\n'
    for i in range(0,self.count()-1):
      w = self.widget(i)
      r += w.export()
      r += self.id + '.insertTab('+str(i)+', '+w.id+')\n'
      r += self.id + '.setTabText('+str(i)+', \''+self.tabText(i)+'\')\n'
    r += self.id + '.setCurrentIndex('+str(self.currentIndex())+')\n'

    return r

