import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
TableView_0 = pycinema.filters.TableView()
ImageView_0 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
TableView_0.inputs.table.set(ImageReader_0.outputs.images, False)
TableView_0.inputs.selection.set([1, 2, 3, 4, 5, 6], False)
ImageView_0.inputs.images.set(TableView_0.outputs.table, False)
ImageView_0.inputs.selection.set([], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setVerticalOrientation()
view3 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame2.insertView( 0, view3 )
view5 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 1, view5 )
splitFrame2.setSizes([339, 474])
splitFrame0.insertView( 1, splitFrame2 )
splitFrame0.setSizes([754, 754])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
tabFrame0.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
