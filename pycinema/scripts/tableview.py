import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

# this application settings
TABLEVIEW = { 'VERSION' : '1.0'}

# args
database = PYCINEMA_ARG_0

# filters
TableReader_0 = pycinema.filters.TableReader()
TableView_0 = pycinema.filters.TableView()

# properties
TableReader_0.inputs.path.set(database, False)
TableReader_0.inputs.file_column.set("FILE", False)
TableView_0.inputs.table.set(TableReader_0.outputs.table, False)
TableView_0.inputs.selection.set([], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame0.insertView( 0, view2 )
splitFrame0.setSizes([1018])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
tabFrame0.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
TableReader_0.update()
