import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.2.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf1 = vf0.insertFrame(1)
vf1.setVerticalOrientation()
ImageView_0 = vf1.insertView( 0, pycinema.theater.views.ImageView() )
ImageView_1 = vf1.insertView( 1, pycinema.theater.views.ImageView() )
vf1.setSizes([493, 493])
vf0.setSizes([1021, 571])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageConvertGrayscale_0 = pycinema.filters.ImageConvertGrayscale()
TableQuery_0 = pycinema.filters.TableQuery()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageConvertGrayscale_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_1.inputs.images.set(ImageConvertGrayscale_0.outputs.images, False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input where phi=-180", False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)

# execute pipeline
CinemaDatabaseReader_0.update()
