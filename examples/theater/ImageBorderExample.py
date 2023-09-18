import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '1.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setVerticalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf1 = vf0.insertFrame(1)
vf1.setHorizontalOrientation()
ImageView_0 = vf1.insertView( 0, pycinema.theater.views.ImageView() )
ImageView_1 = vf1.insertView( 1, pycinema.theater.views.ImageView() )
vf1.setSizes([509, 508])
vf0.setSizes([421, 421])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
ImageBorder_0 = pycinema.filters.ImageBorder()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input where phi=-180", False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageBorder_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageBorder_0.inputs.width.set(10, False)
ImageBorder_0.inputs.color.set("lightblue", False)
ImageView_1.inputs.images.set(ImageBorder_0.outputs.images, False)

# execute pipeline
CinemaDatabaseReader_0.update()
