import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.0.1'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableWriter_0 = pycinema.filters.TableWriter()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableWriter_0.inputs.path.set("TableWriteHeadlessExample.csv", False)
TableWriter_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)

# execute pipeline
CinemaDatabaseReader_0.update()
