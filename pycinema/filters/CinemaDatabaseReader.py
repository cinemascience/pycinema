from pycinema import Filter

import csv
import os.path
import re
import logging as log

from urllib.request import urlopen
from urllib.parse import urljoin

class CinemaDatabaseReader(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'path': '',
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

        is_url = dbPath.startswith(("http://", "https://"))

        if not is_url:
            dbPath = os.path.expanduser(dbPath)

            if not os.path.exists(dbPath):
                log.error(" CDB not found '" + dbPath + "'")
                self.outputs.table.set([[]])
                return 0

        try:
            if is_url:
                dataCsvPath = urljoin(dbPath.rstrip("/") + "/", "data.csv")

                with urlopen(dataCsvPath) as response:
                    content = response.read().decode("utf-8")

                rows = csv.reader(content.splitlines(), delimiter=",")
                for row in rows:
                    table.append(row)
            else:
                dataCsvPath = os.path.join(dbPath, "data.csv")

                with open(dataCsvPath, "r", newline="") as csvfile:
                    rows = csv.reader(csvfile, delimiter=",")
                    for row in rows:
                        table.append(row)

        except Exception as e:
            log.error(f" Unable to open data.csv: {e}")
            self.outputs.table.set([[]])
            return 0

        # remove empty lines
        table = [row for row in table if len(row) > 0]

        # add dbPath prefix to file columns
        fileColumnIndices = [
            i
            for i, item in enumerate(table[0])
            if re.search(self.inputs.file_column.get(), item, re.IGNORECASE)
        ]

        for i in range(1, len(table)):
            for j in fileColumnIndices:
                if (
                    not table[i][j].startswith("http://")
                    and not table[i][j].startswith("https://")
                ):
                    if is_url:
                        table[i][j] = urljoin(
                            dbPath.rstrip("/") + "/",
                            table[i][j]
                        )
                    else:
                        table[i][j] = os.path.join(
                            dbPath,
                            table[i][j]
                        )

        # add id column
        if "id" not in table[0]:
            table[0].append("id")
            for i in range(1, len(table)):
                table[i].append(i - 1)

        self.outputs.table.set(table)

        return 1
