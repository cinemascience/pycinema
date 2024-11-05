import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.0.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()
TableView_0 = pycinema.filters.TableView()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set(TableView_0.inputs.selection, False)
TableView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableView_0.inputs.selection.set([1, 2, 3, 4, 5, 6], False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame1.setSizes([1018])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame2.insertView( 0, view2 )
view3 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 1, view3 )
splitFrame2.setSizes([506, 505])
tabFrame1.insertTab(1, splitFrame2)
tabFrame1.setTabText(1, 'Layout 2')
tabFrame1.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
CinemaDatabaseReader_0.update()
