from pycinema import Filter

import os
import logging as log
import yaml

#
# reads and makes available the data structure from 
# a yaml file
#
class YamlFileReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'file': '', 
            'cache': True
          },
          outputs={
            'data': [] 
          }
        )

    def _update(self):

        data = ''
        p = self.inputs.file.get() 
        self.outputs.data.set([])

        if not os.path.exists(p):
            log.error(" file not found: '" + p + "'")

        try:
            results = []
            with open(p, 'r', encoding='utf-8') as textfile:
                data = yaml.safe_load(textfile)
        except:
             log.error(" Unable to open file: '" + p + "'")

        self.outputs.data.set(data)
        return 1
