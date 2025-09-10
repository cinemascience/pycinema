from pycinema import Filter

import os
import logging as log

from pycinema import getTableExtent

class TextFileReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'file': '', 
            'cache': True
          },
          outputs={
            'text': ''
          }
        )

    def _update(self):

        temptext = ''
        p = self.inputs.file.get()

        if not p:
            temptext.append('')

        if not os.path.exists(p):
            log.error(" file not found: '" + p + "'")
            temptext.append('')

        try:
            with open(p, 'r', encoding='utf-8') as textfile:
                file_contents = textfile.read()
                temptext = file_contents
        except:
            log.error(" Unable to open file: '" + p + "'")

        self.outputs.text.set(temptext)
        return 1
