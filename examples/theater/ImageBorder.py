import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '1.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setVerticalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeView() )
vf2 = vf0.insertFrame(1)
vf2.setHorizontalOrientation()
ImageView_2 = vf2.insertView( 0, pycinema.theater.views.ImageView() )
ImageView_3 = vf2.insertView( 1, pycinema.theater.views.ImageView() )
vf2.setSizes([751, 750])
vf0.setSizes([456, 455])

# filters
CinemaDatabaseReader_1 = pycinema.filters.CinemaDatabaseReader()
ImageBorder_0 = pycinema.filters.ImageBorder()
ImageReader_1 = pycinema.filters.ImageReader()
TableQuery_1 = pycinema.filters.TableQuery()

# properties
CinemaDatabaseReader_1.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_1.inputs.file_column.set("file", False)
ImageBorder_0.inputs.images.set(ImageReader_1.outputs.images, False)
ImageBorder_0.inputs.width.set(20, False)
ImageBorder_0.inputs.color.set("AUTO", False)
ImageReader_1.inputs.table.set(TableQuery_1.outputs.table, False)
ImageReader_1.inputs.file_column.set("FILE", False)
ImageReader_1.inputs.cache.set(True, False)
ImageView_2.inputs.images.set(ImageReader_1.outputs.images, False)
TableQuery_1.inputs.table.set(CinemaDatabaseReader_1.outputs.table, False)
TableQuery_1.inputs.sql.set("SELECT * FROM input WHERE phi=-180", False)
ImageView_3.inputs.images.set(ImageBorder_0.outputs.images, False)

# execute pipeline
CinemaDatabaseReader_1.update()
