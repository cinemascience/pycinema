from PySide6 import QtGui, QtCore

class NodeEditorStyle:

  Z_EDGE_LAYER = 400
  Z_NODE_LAYER = 500

  NODE_HEADER_HEIGHT = 20
  NODE_BORDER_WIDTH = 5
  NODE_SEP_WIDTH = 5
  NODE_WIDTH = 200
  NODE_MARGIN = 3
  NODE_SEP_Y = 25
  NODE_PORT_Y = 35
  NODE_PORT_SPACE = 25

  PORT_SIZE = 8
  PORT_BORDER_WIDTH = 2

  COLOR_BASE = None
  COLOR_BASE_T = None
  COLOR_NORMAL_ = None
  COLOR_DISABLED_ = None
  COLOR_BACKGROUND = None
  COLOR_BORDER = None
  COLOR_WIDGET = None
  COLOR_WIDGET_T = None
  COLOR_RED = None
  COLOR_RED_T = None
  COLOR_BLUE = None
  COLOR_BLUE_T = None
  COLOR_NORMAL = None

  def update_theme():

    base_color = QtCore.QCoreApplication.instance().palette().base().color()

    if base_color.red()<100:
      #dark theme
      NodeEditorStyle.COLOR_BASE = QtGui.QColor('#999')
      NodeEditorStyle.COLOR_BASE_T = QtGui.QColor('#666')
      NodeEditorStyle.COLOR_BASE_T.setAlpha(200)
      NodeEditorStyle.COLOR_NORMAL_ = '#ddd'
      NodeEditorStyle.COLOR_DISABLED_ = '#999'
      NodeEditorStyle.COLOR_BACKGROUND = QtGui.QColor('#333438')
      NodeEditorStyle.COLOR_BORDER = QtGui.QColor('#ff0000')
      NodeEditorStyle.COLOR_WIDGET = QtGui.QColor('#505050')
      NodeEditorStyle.COLOR_WIDGET_T = QtGui.QColor('#444')
      NodeEditorStyle.COLOR_WIDGET_T.setAlpha(200)
      NodeEditorStyle.COLOR_RED = QtGui.QColor('#98281b')
      NodeEditorStyle.COLOR_RED_T = QtGui.QColor('#98281b')
      NodeEditorStyle.COLOR_RED_T.setAlpha(200)
      NodeEditorStyle.COLOR_BLUE = QtGui.QColor('#15a3b4')
      NodeEditorStyle.COLOR_BLUE_T = QtGui.QColor('#15a3b4')
      NodeEditorStyle.COLOR_BLUE_T.setAlpha(200)
      NodeEditorStyle.COLOR_NORMAL = QtGui.QColor(NodeEditorStyle.COLOR_NORMAL_)
    else:
      #bright theme
      NodeEditorStyle.COLOR_BASE = QtGui.QColor('#bbb')
      NodeEditorStyle.COLOR_BASE_T = QtGui.QColor('#ccc')
      NodeEditorStyle.COLOR_BASE_T.setAlpha(200)
      NodeEditorStyle.COLOR_NORMAL_ = '#000'
      NodeEditorStyle.COLOR_DISABLED_ = '#aaa'
      NodeEditorStyle.COLOR_BACKGROUND = QtGui.QColor('#aaa')
      NodeEditorStyle.COLOR_BORDER = QtGui.QColor('#ff0000')
      NodeEditorStyle.COLOR_WIDGET = QtGui.QColor('#eee')
      NodeEditorStyle.COLOR_WIDGET_T = QtGui.QColor('#eee')
      NodeEditorStyle.COLOR_WIDGET_T.setAlpha(200)
      NodeEditorStyle.COLOR_RED = QtGui.QColor('#f24940')
      NodeEditorStyle.COLOR_RED_T = QtGui.QColor('#f24940')
      NodeEditorStyle.COLOR_RED_T.setAlpha(200)
      NodeEditorStyle.COLOR_BLUE = QtGui.QColor('#96deec')
      NodeEditorStyle.COLOR_BLUE_T = QtGui.QColor('#96deec')
      NodeEditorStyle.COLOR_BLUE_T.setAlpha(200)
      NodeEditorStyle.COLOR_NORMAL = QtGui.QColor(NodeEditorStyle.COLOR_NORMAL_)
