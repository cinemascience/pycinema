import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf2 = vf0.insertFrame(0)
vf2.setVerticalOrientation()
vf2.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf3 = vf2.insertFrame(1)
vf3.setHorizontalOrientation()
ImageView_4 = vf3.insertView( 0, pycinema.theater.views.ImageView() )
ImageView_5 = vf3.insertView( 1, pycinema.theater.views.ImageView() )
vf3.setSizes([687, 687])
vf2.setSizes([421, 421])
vf0.setSizes([1381])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
Python_1 = pycinema.filters.Python()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
Python_1.inputs.inputs.set(ImageReader_0.outputs.images, False)
Python_1.inputs.code.set("examples/pythonfilter/ImageHistogram.py", False)
ImageView_4.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_5.inputs.images.set(Python_1.outputs.outputs, False)

# execute pipeline
CinemaDatabaseReader_0.update()
