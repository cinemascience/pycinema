from pycinema import Filter, isNumber

import sqlite3

from pycinema import getTableExtent
import logging as log

def queryData(db, sqlQuery):
  c = db.cursor()
  try:
    c.execute(sqlQuery)
  except sqlite3.Error as er:
    log.error(' %s' % (' '.join(er.args)))
    return [[]]
  res = c.fetchall()
  columns = []
  for d in c.description:
    columns.append(d[0])
  res.insert(0,columns)
  return res

class DBQueryToTable(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'db': None, 
            'sql': 'SELECT * FROM input'
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

      output = queryData(self.inputs.db.get(), self.inputs.sql.get())

      self.outputs.table.set(output)

      return 1;
