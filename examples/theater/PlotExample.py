import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.0.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf3 = vf0.insertFrame(0)
vf3.setVerticalOrientation()
vf3.insertView( 0, pycinema.theater.views.NodeEditorView() )
TableView_0 = vf3.insertView( 1, pycinema.theater.views.TableView() )
vf3.setSizes([421, 421])
vf1 = vf0.insertFrame(1)
vf1.setVerticalOrientation()
PlotView_0 = vf1.insertView( 0, pycinema.theater.views.PlotView() )
PlotView_1 = vf1.insertView( 1, pycinema.theater.views.PlotView() )
vf1.setSizes([421, 421])
vf2 = vf0.insertFrame(2)
vf2.setVerticalOrientation()
PlotView_2 = vf2.insertView( 0, pycinema.theater.views.PlotView() )
PlotView_3 = vf2.insertView( 1, pycinema.theater.views.PlotView() )
vf2.setSizes([421, 421])
vf0.setSizes([608, 609, 608])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()

# properties
PlotView_0.inputs.title.set("Fibonacci", False)
PlotView_0.inputs.x_values.set("a", False)
PlotView_0.inputs.y_values.set("b", False)
PlotView_0.inputs.table.set(TableQuery_0.outputs.table, False)
PlotView_1.inputs.title.set("LIne", False)
PlotView_1.inputs.x_values.set("a", False)
PlotView_1.inputs.y_values.set("c", False)
PlotView_1.inputs.table.set(TableQuery_0.outputs.table, False)
PlotView_2.inputs.title.set("Reverse Fibonnaci", False)
PlotView_2.inputs.x_values.set("b", False)
PlotView_2.inputs.y_values.set("a", False)
PlotView_2.inputs.table.set(TableQuery_0.outputs.table, False)
PlotView_3.inputs.title.set("Upside Down", False)
PlotView_3.inputs.x_values.set("b", False)
PlotView_3.inputs.y_values.set("c", False)
PlotView_3.inputs.table.set(TableQuery_0.outputs.table, False)
CinemaDatabaseReader_0.inputs.path.set("data/plot.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input WHERE a>0", False)
TableView_0.inputs.table.set(TableQuery_0.outputs.table, False)

# execute pipeline
PlotView_0.update()
