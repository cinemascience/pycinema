import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.0.1'}

# this application settings
DISCOVER = { 'VERSION' : '1.0'}

# args
database    = PYCINEMA_ARG_0
read_filter = PYCINEMA_ARG_1
selected    = PYCINEMA_ARG_2 

# reporting
print("discover v" + DISCOVER["VERSION"])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableView_0 = pycinema.filters.TableView()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()
ParallelCoordinates_0 = pycinema.filters.ParallelCoordinates()
ImageReader_1 = pycinema.filters.ImageReader()
ImageView_1 = pycinema.filters.ImageView()
ValueSource_0 = pycinema.filters.ValueSource()
TableView_1 = pycinema.filters.TableView()
TableQuery_0 = pycinema.filters.TableQuery()

# properties
CinemaDatabaseReader_0.inputs.path.set(database, False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableView_0.inputs.table.set(TableQuery_0.outputs.table, False)
TableView_0.inputs.selection.set(ValueSource_0.outputs.value, False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set(ValueSource_0.outputs.value, False)
ParallelCoordinates_0.inputs.table.set(TableQuery_0.outputs.table, False)
ParallelCoordinates_0.inputs.ignore.set(['^file', '^id'], False)
ParallelCoordinates_0.inputs.selection.set(ValueSource_0.outputs.value, False)
ImageReader_1.inputs.table.set(TableView_0.outputs.tableSelection, False)
ImageReader_1.inputs.file_column.set("FILE", False)
ImageReader_1.inputs.cache.set(True, False)
ImageView_1.inputs.images.set(ImageReader_1.outputs.images, False)
ImageView_1.inputs.selection.set([], False)
ValueSource_0.inputs.value.set(selected, False)
TableView_1.inputs.table.set(TableView_0.outputs.tableSelection, False)
TableView_1.inputs.selection.set([], False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set(read_filter, False)

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
splitFrame3 = pycinema.theater.SplitFrame()
splitFrame3.setVerticalOrientation()
splitFrame4 = pycinema.theater.SplitFrame()
splitFrame4.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame4.insertView( 0, view2 )
splitFrame4.setSizes([335])
splitFrame3.insertView( 0, splitFrame4 )
view3 = pycinema.theater.views.FilterView( ParallelCoordinates_0 )
splitFrame3.insertView( 1, view3 )
splitFrame3.setSizes([463, 350])
splitFrame2.insertView( 0, splitFrame3 )
view4 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 1, view4 )
splitFrame5 = pycinema.theater.SplitFrame()
splitFrame5.setVerticalOrientation()
view5 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame5.insertView( 0, view5 )
view6 = pycinema.theater.views.FilterView( TableView_1 )
splitFrame5.insertView( 1, view6 )
splitFrame5.setSizes([557, 256])
splitFrame2.insertView( 2, splitFrame5 )
splitFrame2.setSizes([335, 334, 335])
tabFrame1.insertTab(1, splitFrame2)
tabFrame1.setTabText(1, 'Layout 2')
tabFrame1.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
CinemaDatabaseReader_0.update()
