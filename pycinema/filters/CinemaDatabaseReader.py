from pycinema import Filter

import csv

class CinemaDatabaseReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': '',
            'file_column': 'file'
          },
          outputs={
            'table': [[]]
          }
        )

    def _update(self):

        table = [];
        dbPath = self.inputs.path.get();
        dataCsvPath = dbPath + '/data.csv';
        with open(dataCsvPath, 'r+') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            for row in rows:
                table.append(row)

        # remove empty lines
        table = list(filter(lambda row: len(row)>0, table))

        # force lower case header
        table[0] = list(map(str.lower,table[0]))

        # add dbPath prefix to file column
        fileColumnIdx = table[0].index( self.inputs.file_column.get() );
        for i in range(1,len(table)):
            table[i][fileColumnIdx] = dbPath + '/' + table[i][fileColumnIdx];

        self.outputs.table.set(table);

        return 1;
