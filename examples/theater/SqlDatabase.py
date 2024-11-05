import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.0.0'}

# filters
SqliteDatabaseReader_0 = pycinema.filters.SqliteDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()

# properties
SqliteDatabaseReader_0.inputs.path.set("data/scalar-images.cdb/data.db", False)
SqliteDatabaseReader_0.inputs.table.set("data", False)
SqliteDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(SqliteDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
view3 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame1.insertView( 1, view3 )
splitFrame1.setSizes([857, 855])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
tabFrame1.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
SqliteDatabaseReader_0.update()
