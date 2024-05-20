from pycinema import Filter

import sqlite3

from os import path
import re
import logging as log

class SqliteDatabaseReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': '',
            'table': '',
            'file_column': 'FILE'
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

        table = []
        dbPath = self.inputs.path.get()
        if not dbPath:
            self.outputs.table.set([[]])
            return 0

        if not path.exists(dbPath):
            log.error(" sqlite db not found: '" + dbPath + "'")
            self.outputs.table.set([[]])
            return 0

        tname = self.inputs.table.get()
        try:
            conn = sqlite3.connect(dbPath)
            cursor = conn.cursor()

            # capture the names of each column
            cdata = cursor.execute(f'PRAGMA table_info({tname});').fetchall()

            cnames = [entry[1] for entry in cdata]
            table.append(cnames)

            # capture row data
            data = cursor.execute("SELECT * FROM " + tname + "").fetchall() #LIMIT 10
            for row in data:
                # tuple output convert to list
                table.append(list(row))

            cursor.close()

        except sqlite3.Error as error:
            log.error(" Error while connecting to sqlite: " + error)
            self.outputs.table.set([[]])
            return 0
        finally:
            if conn:
                conn.close()

        # remove empty lines
        table = list(filter(lambda row: len(row)>0, table))

        # add dbPath prefix to file column
        try:
            fileColumnIdx = [i for i, item in enumerate(table[0]) if re.search(self.inputs.file_column.get(), item, re.IGNORECASE)].pop()
        except:
            log.error(" file column not found: '" + self.inputs.file_column.get() + "'")
            self.outputs.table.set([[]])
            return 0

        dbPathPrefix = path.dirname(dbPath)
        for i in range(1,len(table)):
            if not table[i][fileColumnIdx].startswith('http:') and not table[i][fileColumnIdx].startswith('https'):
                table[i][fileColumnIdx] = dbPathPrefix + '/' + table[i][fileColumnIdx]

        self.outputs.table.set(table)

        return 1
