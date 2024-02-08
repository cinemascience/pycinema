from pycinema import Filter

import csv
from os.path import exists
import re
import logging as log

class TextFileSource(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': ''
          },
          outputs={
            'text': ''
          }
        )

    def _update(self):

        filePath = self.inputs.path.get()

        if not filePath:
            self.outputs.text.set("")
            return 0

        if not exists(filePath):
            log.error(" file not found: '" + filePath + "'")
            self.outputs.text.set("")
            return 0

        try:
            with open(filePath, 'r+') as textfile:
                self.outputs.text.set(textfile.read())
        except:
            log.error(" Unable to open file: '" + filePath + "'")
            self.outputs.text.set([[]])
            return 0
        return 1
