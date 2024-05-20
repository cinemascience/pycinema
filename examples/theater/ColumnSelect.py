import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.0.2'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()
TableView_0 = pycinema.filters.TableView()
ValueSource_0 = pycinema.filters.ValueSource()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set(ValueSource_0.outputs.value, False)
TableView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableView_0.inputs.selection.set(ValueSource_0.outputs.value, False)
ValueSource_0.inputs.value.set([0, 1, 2, 3, 4, 5, 6], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame0.setSizes([1493])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view4 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame1.insertView( 0, view4 )
view3 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame1.insertView( 1, view3 )
splitFrame1.setSizes([744, 742])
tabFrame0.insertTab(1, splitFrame1)
tabFrame0.setTabText(1, 'Layout 2')
tabFrame0.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
