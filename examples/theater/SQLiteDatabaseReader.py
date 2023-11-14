import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.0.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeEditorView() )
ImageView_0 = vf0.insertView( 1, pycinema.theater.views.ImageView() )
vf0.setSizes([703, 701])

# filters
SqliteDatabaseReader_0 = pycinema.filters.SqliteDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()

# properties
SqliteDatabaseReader_0.inputs.path.set("data/wildfire/wildfire.sqlite3", False)
SqliteDatabaseReader_0.inputs.table.set("vision", False)
SqliteDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(SqliteDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)

# execute pipeline
SqliteDatabaseReader_0.update()
