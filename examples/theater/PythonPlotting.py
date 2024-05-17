import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
Calculator_0 = pycinema.filters.Calculator()
TableQuery_0 = pycinema.filters.TableQuery()
Python_0 = pycinema.filters.Python()
TableView_0 = pycinema.filters.TableView()
ImageView_0 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/scalar-images.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
Calculator_0.inputs.table.set(TableQuery_0.outputs.table, False)
Calculator_0.inputs.label.set("result", False)
Calculator_0.inputs.expression.set("time*id", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input", False)
Python_0.inputs.inputs.set(Calculator_0.outputs.table, False)
Python_0.inputs.code.set("examples/pythonfilter/PythonPlotting.py", False)
TableView_0.inputs.table.set(Calculator_0.outputs.table, False)
TableView_0.inputs.selection.set([], False)
ImageView_0.inputs.images.set(Python_0.outputs.outputs, False)
ImageView_0.inputs.selection.set([], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame0.setSizes([1295])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
view4 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame2.insertView( 0, view4 )
view6 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 1, view6 )
splitFrame2.setSizes([768, 767])
tabFrame0.insertTab(1, splitFrame2)
tabFrame0.setTabText(1, 'Layout 3')
tabFrame0.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
