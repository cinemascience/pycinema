import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setVerticalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf1 = vf0.insertFrame(1)
vf1.setHorizontalOrientation()
TableView_0 = vf1.insertView( 0, pycinema.theater.views.TableView() )
ImageView_1 = vf1.insertView( 1, pycinema.theater.views.ImageView() )
vf1.setSizes([636, 735])
vf0.setSizes([462, 461])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()

# properties
CinemaDatabaseReader_0.inputs.path.set(PYCINEMA_ARG_0, False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set(PYCINEMA_ARG_1, False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
TableView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableView_0.inputs.selection.set([], False)
ImageView_1.inputs.images.set(ImageReader_0.outputs.images, False)

# execute pipeline
CinemaDatabaseReader_0.update()
