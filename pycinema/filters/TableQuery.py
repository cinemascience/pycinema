from pycinema import Filter, isNumber

import sqlite3

from pycinema import getTableExtent

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

    def executeSQL(self,db,sql):
        try:
            c = db.cursor()
            c.execute(sql)
        except sqlite3.Error as e:
            print(e)

    def createTable(self, db, table):
        sql = 'CREATE TABLE input(id INTEGER PRIMARY KEY AUTOINCREMENT';

        header = table[0]
        firstRow = table[1]

        for i in range(0,len(header)):
            if header[i].lower()=='id':
                continue
            if isNumber(firstRow[i]):
                sql += ', ' + header[i] + ' REAL';
            else:
                sql += ', ' + header[i] + ' TEXT';
        sql =  sql + ')';
        self.executeSQL(db,sql)

    def insertData(self, db, table):
        sql = 'INSERT INTO input(';
        for x in table[0]:
            sql = sql + x + ', ';
        sql = sql[0:-2] + ') VALUES\n';

        for i in range(1, len(table)):
            row = '('
            for v in table[i]:
              row += '"' + str(v) + '",'
            sql += row[0:-1] + '),\n'
        sql = sql[0:-2];
        self.executeSQL(db,sql)

    def queryData(self, db, sqlQuery):
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

    def _update(self):

      db = sqlite3.connect(":memory:")

      table = self.inputs.table.get()
      tableExtent = getTableExtent(table)
      if tableExtent[0]<2 or tableExtent[1]<1:
          return self.outputs.table.set([[]])

      self.createTable(db, table)
      self.insertData(db, table)

      sql = self.inputs.sql.get()
      if isinstance(sql, dict):
        header = table[0]
        sql2 = {k: v for k, v in sql.items() if k in header}
        sql = 'SELECT * FROM input WHERE '+ ' AND '.join(sql2.values())

      output = self.queryData(db, sql)

      self.outputs.table.set(output)

      return 1;
