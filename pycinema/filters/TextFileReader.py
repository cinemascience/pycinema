from pycinema import Filter

import csv
import os
import re
import logging as log

from pycinema import getTableExtent

class TextFileSource(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'table': [[]],
            'file_column': 'FILE',
            'cache': True
          },
          outputs={
            'text': ''
          }
        )

    def _update(self):

        table = self.inputs.table.get()
        tableExtent = getTableExtent(table)
        fileColumn = self.inputs.file_column.get()

        try:
            fileColumnIdx = [i for i, item in enumerate(table[0]) if re.search(fileColumn, item, re.IGNORECASE)].pop()
        except Exception as e:
            log.error("table does not contain '" + fileColumn + "' column!")
            return 0

        text = ''
        for i in range(1, len(table)):

            row = table[i]
            filePath = row[fileColumnIdx]

            if not filePath:
                text = text

            if not os.path.exists(filePath):
                log.error(" file not found: '" + filePath + "'")
                text = text

            try:
                with open(filePath, 'r') as textfile:
                    text = text + '\n' + filePath + '\n' + textfile.read()
            except:
                log.error(" Unable to open file: '" + filePath + "'")
                text = text
        self.outputs.text.set(text)
        return 1
