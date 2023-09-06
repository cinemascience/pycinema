from PySide6 import QtGui, QtCore

class ViewStyle:

  COLOR_SPLITTER = 'grey'

  def update_theme():

    base_color = QtCore.QCoreApplication.instance().palette().base().color()

    if base_color.red()<100:
      #dark theme
      ViewStyle.COLOR_SPLITTER = 'black'

    else:
      #bright theme
      ViewStyle.COLOR_SPLITTER = 'white'

  def get_style_sheet():
      return "QSplitter::handle {background-color: " + ViewStyle.COLOR_SPLITTER + ";}"

