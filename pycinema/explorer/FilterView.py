from PySide6 import QtCore, QtWidgets, QtGui

from .View import *

class FilterView(View):

    def __init__(self, filter=filter):
        super().__init__()

        self.filter = filter()
        self.setTitle(self.filter.id)
        self.filter.inputs.container.set(self.content)
