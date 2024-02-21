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
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame1.setSizes([1018])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
splitFrame3 = pycinema.theater.SplitFrame()
splitFrame3.setVerticalOrientation()
view5 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame3.insertView( 0, view5 )
view7 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame3.insertView( 1, view7 )
splitFrame3.setSizes([339, 474])
tabFrame1.insertTab(1, splitFrame3)
tabFrame1.setTabText(1, 'Layout 2')
tabFrame1.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
CinemaDatabaseReader_0.update()
