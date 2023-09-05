from PySide6 import QtCore,QtGui

class Icons:

  def update_theme():
    base_color = QtCore.QCoreApplication.instance().palette().base().color()
    if base_color.red()<100:
      icon_color = '#999'
    else:
      icon_color = '#666'

    Icons.icon_close = '<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path style="fill:'+icon_color+'" d="m256-200-56-56 224-224-224-224 56-56 224 224 224-224 56 56-224 224 224 224-56 56-224-224-224 224Z"/></svg>'
    Icons.icon_split_h = '<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path style="fill:'+icon_color+'" d="M600-120q-33 0-56.5-23.5T520-200v-560q0-33 23.5-56.5T600-840h160q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H600Zm-400 0q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h160q33 0 56.5 23.5T440-760v560q0 33-23.5 56.5T360-120H200Zm0-640v560h160v-560H200Zm160 560H200h160Z"/></svg>'
    Icons.icon_split_v = '<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path style="fill:'+icon_color+'" d="M200-120q-33 0-56.5-23.5T120-200v-160q0-33 23.5-56.5T200-440h560q33 0 56.5 23.5T840-360v160q0 33-23.5 56.5T760-120H200Zm0-400q-33 0-56.5-23.5T120-600v-160q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v160q0 33-23.5 56.5T760-520H200Zm560-240H200v160h560v-160ZM200-600v-160 160Z"/></svg>'

  def toQIcon(svg_str):
    svg_bytes = bytearray(svg_str, encoding='utf-8')
    return QtGui.QIcon(
      QtGui.QPixmap.fromImage(
        QtGui.QImage.fromData(svg_bytes)
      )
    )
