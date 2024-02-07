import pycinema
import pycinema.filters

# pycinema settings
PYCINEMA = { 'VERSION' : '1.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableWriter_0 = pycinema.filters.TableWriter()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableWriter_0.inputs.path.set("TableWriteExample.csv", False)
TableWriter_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)

# execute pipeline
CinemaDatabaseReader_0.update()
