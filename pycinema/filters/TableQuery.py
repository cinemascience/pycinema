from pycinema import Filter, isNumber

import sqlite3

from pycinema import getTableExtent

def executeSQL(db,sql):
  try:
    c = db.cursor()
    c.execute(sql)
  except sqlite3.Error as e:
    print(e)

def createTable(db, table, castType=False):
  sql = 'CREATE TABLE input(id INTEGER PRIMARY KEY AUTOINCREMENT';
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
  sql = 'INSERT INTO input(';
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

def queryData(db, sqlQuery):
  c = db.cursor()
  try:
    c.execute(sqlQuery)
  except sqlite3.Error as er:
    print('[SQL ERROR] %s' % (' '.join(er.args)))
    return [[]]
  res = c.fetchall()
  columns = []
  for d in c.description:
    columns.append(d[0])
  res.insert(0,columns)
  return res

class TableQuery(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'table': [[]],
            'sql': 'SELECT * FROM input'
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

      db = sqlite3.connect(":memory:")

      table = self.inputs.table.get()
      tableExtent = getTableExtent(table)
      if tableExtent[0]<2 or tableExtent[1]<1:
          return self.outputs.table.set([[]])

      createTable(db, table, True)
      insertData(db, table)

      sql = self.inputs.sql.get()
      if isinstance(sql, dict):
        header = table[0]
        sql2 = {k: v for k, v in sql.items() if k in header}
        sql = 'SELECT * FROM input WHERE '+ ' AND '.join(sql2.values())

      output = queryData(db, sql)

      self.outputs.table.set(output)

      return 1;
