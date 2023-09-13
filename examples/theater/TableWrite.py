import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '1.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeView() )
vf0.setSizes([1024])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableWriter_0 = pycinema.filters.TableWriter()

# properties
CinemaDatabaseReader_0.inputs.path.set("../data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("file", False)
TableWriter_0.inputs.path.set("test.csv", False)
TableWriter_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)

# execute pipeline
CinemaDatabaseReader_0.update()
