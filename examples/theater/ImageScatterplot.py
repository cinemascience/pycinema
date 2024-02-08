import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf1 = vf0.insertFrame(1)
vf1.setVerticalOrientation()
ImageView_1 = vf1.insertView( 0, pycinema.theater.views.ImageView() )
ImageView_2 = vf1.insertView( 1, pycinema.theater.views.ImageView() )
vf1.setSizes([366, 476])
vf0.setSizes([989, 989])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImagesToTable_0 = pycinema.filters.ImagesToTable()
Python_0 = pycinema.filters.Python()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_1.inputs.images.set(ImageReader_0.outputs.images, False)
ImagesToTable_0.inputs.images.set(ImageReader_0.outputs.images, False)
Python_0.inputs.inputs.set(ImagesToTable_0.outputs.table, False)
Python_0.inputs.code.set("examples/pythonfilter/ImageScatterplot.py", False)
ImageView_2.inputs.images.set(Python_0.outputs.outputs, False)

# execute pipeline
CinemaDatabaseReader_0.update()
