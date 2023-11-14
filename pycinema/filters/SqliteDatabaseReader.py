from pycinema import Filter

import sqlite3

from os.path import exists
import re

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

        if not exists(dbPath):
            print('[ERROR] sqlite db not found:', dbPath)
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

            # print(table)
            # capture row data
            data = cursor.execute("SELECT * FROM " + tname + "").fetchall() #LIMIT 10
            for row in data:
                # tuple output convert to list
                table.append(list(row))

            # print(table)
            cursor.close()
        except sqlite3.Error as error:
            print("Error while connecting to sqlite", error)
            self.outputs.table.set([[]])
            return 0
        finally:
            if conn:
                conn.close()

        # remove empty lines
        table = list(filter(lambda row: len(row)>0, table))

        # # force lower case header
        # table[0] = list(map(str.lower,table[0]))

        try:
            fileColumnIdx = [i for i, item in enumerate(table[0]) if re.search(self.inputs.file_column.get(), item, re.IGNORECASE)].pop()
        except:
            print('[ERROR] file column not found:',self.inputs.file_column.get())
            self.outputs.table.set([[]])
            return 0

        for i in range(1,len(table)):
            table[i][fileColumnIdx] = table[i][fileColumnIdx]

        self.outputs.table.set(table)

        return 1
