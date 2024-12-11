import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

# filters
TableReader_0 = pycinema.filters.TableReader()
TableReader_1 = pycinema.filters.TableReader()
TableQuery_0 = pycinema.filters.TableQuery()
TableQuery_1 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
ImageReader_1 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()
ImageView_1 = pycinema.filters.ImageView()

# properties
TableReader_0.inputs.path.set("data/sphere.cdb", False)
TableReader_0.inputs.file_column.set("FILE", False)
TableReader_1.inputs.path.set("data/wildfire/wildfiredataSmall.csv", False)
TableReader_1.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(TableReader_1.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input LIMIT 10", False)
TableQuery_1.inputs.table.set(TableReader_0.outputs.table, False)
TableQuery_1.inputs.sql.set("SELECT * FROM input LIMIT 12", False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set(TableReader_1.inputs.file_column, False)
ImageReader_0.inputs.cache.set(True, False)
ImageReader_1.inputs.table.set(TableQuery_1.outputs.table, False)
ImageReader_1.inputs.file_column.set(TableReader_0.inputs.file_column, False)
ImageReader_1.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageView_1.inputs.images.set(ImageReader_1.outputs.images, False)
ImageView_1.inputs.selection.set([], False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setVerticalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 0, view2 )
view3 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame2.insertView( 1, view3 )
splitFrame2.setSizes([506, 505])
splitFrame1.insertView( 1, splitFrame2 )
splitFrame1.setSizes([321, 492])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
tabFrame1.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
TableReader_0.update()
