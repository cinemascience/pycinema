import pycinema
import pycinema.filters
from pathlib import Path

# pycinema settings
PYCINEMA = { 'VERSION' : '3.0.1'}

results = "scratch/TableWriteExample.csv"
print("Test script that reads a cinema database and writes a table")
print("  writing to " + results)

Path("scratch").mkdir(parents=True, exist_ok=True)

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableWriter_0 = pycinema.filters.TableWriter()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableWriter_0.inputs.path.set(results, False)
TableWriter_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)

# execute pipeline
CinemaDatabaseReader_0.update()
