import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
SqliteDatabaseReader_0 = pycinema.filters.SqliteDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()

# properties
SqliteDatabaseReader_0.inputs.path.set("data/wildfire/wildfire.sqlite3", False)
SqliteDatabaseReader_0.inputs.table.set("vision", False)
SqliteDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(SqliteDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame0.setSizes([1018])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame1.insertView( 0, view2 )
splitFrame1.setSizes([1018])
tabFrame0.insertTab(1, splitFrame1)
tabFrame0.setTabText(1, 'Layout 2')
tabFrame0.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
SqliteDatabaseReader_0.update()
