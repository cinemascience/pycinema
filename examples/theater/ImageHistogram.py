import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf1 = vf0.insertFrame(0)
vf1.setVerticalOrientation()
vf1.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf2 = vf1.insertFrame(1)
vf2.setHorizontalOrientation()
ImageView_0 = vf2.insertView( 0, pycinema.theater.views.ImageView() )
ImageView_1 = vf2.insertView( 1, pycinema.theater.views.ImageView() )
vf2.setSizes([509, 508])
vf1.setSizes([421, 421])
vf0.setSizes([1024])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
Python_0 = pycinema.filters.Python()

# properties
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_1.inputs.images.set(Python_0.outputs.outputs, False)
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
Python_0.inputs.inputs.set(ImageReader_0.outputs.images, False)
Python_0.inputs.code.set("examples/pythonfilter/ImageHistogram.py", False)

# execute pipeline
ImageView_0.update()
