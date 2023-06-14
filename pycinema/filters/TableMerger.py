from pycinema import Filter

import sqlite3

class TableMerger(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'table0': [[]],
            'table1': [[]]
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

    def runSQL(self, db, sql):
        c = db.cursor();
        c.execute(sql);
        res = c.fetchall();
        columns = [];
        for d in c.description:
            columns.append(d[0]);
        res.insert(0,columns);
        return res;

    def createTable(self, db, tableIdx, table):
        sql = 'CREATE TABLE table'+str(tableIdx)+'(id INTEGER PRIMARY KEY AUTOINCREMENT';

        header = table[0]
        firstRow = table[1]

        for i in range(0,len(header)):
            if header[i].lower()=='id':
                continue

            sql = sql + ', ' + header[i];
            if isNumber(firstRow[i]):
                sql = sql + ' REAL';
            else:
                sql = sql + ' TEXT';

        sql =  sql + ')';
        self.executeSQL(db,sql)

    def insertData(self, db, tableIdx, table):
        sql = 'INSERT INTO table'+str(tableIdx)+'(';
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
        c = db.cursor();
        c.execute(sqlQuery);
        res = c.fetchall();
        columns = [];
        for d in c.description:
            columns.append(d[0]);
        res.insert(0,columns);
        return res;

    def _update(self):

      db = sqlite3.connect(":memory:");

      tables = [self.inputs.table0.get(),self.inputs.table1.get()]
      nTables = len(tables)

      if len(tables[0])<1 or len(tables[1])<1:
          return 0

      for i in range(0,len(tables)):
          self.createTable(db, i, tables[i]);
          self.insertData(db, i, tables[i]);

      # find shared columns
      column_in_tables = {}
      for i in range(0,nTables):
          header = tables[i][0]
          for j in header:
              column_in_tables[j] = 0
      for i in range(0,nTables):
          header = tables[i][0]
          for j in header:
              column_in_tables[j] += 1

      shared_columns = []
      for key in column_in_tables:
          if column_in_tables[key]==nTables and key!='file':
              shared_columns.append(key)

      sql = 'SELECT \n'
      for i in range(0,nTables):
          header = tables[i][0]
          for j in header:
              if j not in shared_columns and j!='file':
                  sql += '  table'+str(i)+'.'+j+','
      sql += '\n'
      for key in shared_columns:
          sql += '  table0.'+key+','
      sql = sql[0:-1] # remove last comma
      sql += '\nFROM table0 FULL OUTER JOIN table1 ON \n'
      for key in shared_columns:
          sql += '  table0.'+key+"="+'table1.'+key +' AND \n'

      sql = sql[0:-6] # remove last suffix
      output = self.runSQL(db, sql);
      self.outputs.table.set(output);

      return 1;
