import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

# this application settings
DISCOVER = { 'VERSION' : '1.1'}

# args
database    = PYCINEMA_ARG_0
read_filter = "SELECT * FROM input LIMIT 100" 

# reporting
print("discover v" + DISCOVER["VERSION"])
print("    limiting input using query:\'" + read_filter + "\'")
print("    change this by editing input of TableQuery filter")

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
TableView_0 = pycinema.filters.TableView()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_1 = pycinema.filters.ImageReader()
ParallelCoordinates_0 = pycinema.filters.ParallelCoordinates()
ImageView_0 = pycinema.filters.ImageView()
ImageView_1 = pycinema.filters.ImageView()
TableView_1 = pycinema.filters.TableView()

# properties
CinemaDatabaseReader_0.inputs.path.set(database, False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
TableView_0.inputs.table.set(TableQuery_0.outputs.table, False)
TableView_0.inputs.selection.set(ParallelCoordinates_0.inputs.selection, False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set(read_filter, False)
ImageReader_1.inputs.table.set(TableView_0.outputs.table, False)
ImageReader_1.inputs.file_column.set("FILE", False)
ImageReader_1.inputs.cache.set(True, False)
ParallelCoordinates_0.inputs.table.set(TableQuery_0.outputs.table, False)
ParallelCoordinates_0.inputs.ignore.set(['^file', '^id'], False)
ParallelCoordinates_0.inputs.selection.set([1,2,3,4], False)
ParallelCoordinates_0.inputs.compose.set("", False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set(TableView_0.inputs.selection, False)
ImageView_1.inputs.images.set(ImageReader_1.outputs.images, False)
ImageView_1.inputs.selection.set([], False)
TableView_1.inputs.table.set(TableView_0.outputs.table, False)
TableView_1.inputs.selection.set([], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame0.setSizes([2446])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setVerticalOrientation()
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
splitFrame3 = pycinema.theater.SplitFrame()
splitFrame3.setVerticalOrientation()
view2 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame3.insertView( 0, view2 )
view7 = pycinema.theater.views.FilterView( ParallelCoordinates_0 )
splitFrame3.insertView( 1, view7 )
splitFrame3.setSizes([504, 508])
splitFrame2.insertView( 0, splitFrame3 )
view11 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 1, view11 )
splitFrame4 = pycinema.theater.SplitFrame()
splitFrame4.setVerticalOrientation()
view12 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame4.insertView( 0, view12 )
view13 = pycinema.theater.views.FilterView( TableView_1 )
splitFrame4.insertView( 1, view13 )
splitFrame4.setSizes([504, 508])
splitFrame2.insertView( 2, splitFrame4 )
splitFrame2.setSizes([783, 931, 718])
splitFrame1.insertView( 0, splitFrame2 )
splitFrame1.setSizes([1019])
tabFrame0.insertTab(1, splitFrame1)
tabFrame0.setTabText(1, 'Layout 2')
tabFrame0.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
