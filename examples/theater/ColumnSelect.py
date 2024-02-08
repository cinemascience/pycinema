import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setVerticalOrientation()
vf2 = vf0.insertFrame(0)
vf2.setHorizontalOrientation()
vf2.insertView( 0, pycinema.theater.views.NodeEditorView() )
ImageView_1 = vf2.insertView( 1, pycinema.theater.views.ImageView() )
vf2.setSizes([761, 761])
vf1 = vf0.insertFrame(1)
vf1.setHorizontalOrientation()
TableView_0 = vf1.insertView( 0, pycinema.theater.views.TableView() )
ImageView_2 = vf1.insertView( 1, pycinema.theater.views.ImageView() )
vf1.setSizes([762, 760])
vf0.setSizes([476, 476])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
TableView_0.inputs.table.set(ImageReader_0.outputs.images, False)
TableView_0.inputs.selection.set([2, 3, 4, 5, 6, 7], False)
ImageView_1.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_2.inputs.images.set(TableView_0.outputs.table, False)

# execute pipeline
CinemaDatabaseReader_0.update()
