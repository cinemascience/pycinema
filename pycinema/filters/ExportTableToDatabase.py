from pycinema import Filter, isNumber

import sqlite3

from pycinema import getTableExtent
import logging as log

def executeSQL(db,sql):
  try:
    c = db.cursor()
    c.execute(sql)
    db.commit()
  except sqlite3.Error as e:
    log.error(" sqlite3 error: " + e)

def createTable(db, table, tablename, castType=False):
  sql = 'CREATE TABLE '  + tablename + ' (id INTEGER PRIMARY KEY AUTOINCREMENT';
  header = table[0]
  firstRow = table[1]

  if castType:
    for i in range(0,len(header)):
      if header[i].lower()=='id':
        continue
      if isNumber(firstRow[i]):
          sql += ', "' + header[i] + '" REAL';
      else:
          sql += ', "' + header[i] + '" TEXT';
  else:
    for i in range(0,len(header)):
      if header[i].lower()=='id':
        continue
      sql = sql + ', "' + header[i] + '" TEXT';

  sql =  sql + ')';
  executeSQL(db,sql)

def insertData(db, table):
  sql = 'INSERT INTO datacsv (';
  for x in table[0]:
    sql = sql+ '"' + x + '", ';
  sql = sql[0:-2] + ') VALUES\n';

  for i in range(1, len(table)):
    row = '('
    for v in table[i]:
      row += '"' + str(v) + '",'
    sql += row[0:-1] + '),\n'
  sql = sql[0:-2];
  executeSQL(db,sql)

class ExportTableToDatabase(Filter):

    def __init__(self): super().__init__(
          inputs={
            'table': [[]],
            'tablename' : 'datacsv',
            'path': 'default.db'
          },
          outputs={
          }
        )

    def _update(self):

      db = sqlite3.connect(":memory:")

      table = self.inputs.table.get()
      tableExtent = getTableExtent(table)
      if tableExtent[0]<2 or tableExtent[1]<1:
          return self.outputs.table.set([[]])

      createTable(db, table, self.inputs.tablename.get(), True)
      insertData(db, table)

      # save the database to a file
      file_conn = sqlite3.connect(self.inputs.path.get())
      db.backup(file_conn)
      file_conn.close()
      db.close()

      return 1;
