import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.0.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf1 = vf0.insertFrame(0)
vf1.setVerticalOrientation()
vf1.insertView( 0, pycinema.theater.views.NodeEditorView() )
TableView_0 = vf1.insertView( 1, pycinema.theater.views.TableView() )
vf1.setSizes([421, 421])
vf2 = vf0.insertFrame(1)
vf2.setVerticalOrientation()
PlotLineView_0 = vf2.insertView( 0, pycinema.theater.views.PlotLineView() )
PlotBarView_0 = vf2.insertView( 1, pycinema.theater.views.PlotBarView() )
vf2.setSizes([421, 421])
vf0.setSizes([908, 906])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
PlotBarItem_0 = pycinema.filters.PlotBarItem()

# properties
TableView_0.inputs.table.set(TableQuery_0.outputs.table, False)
PlotLineView_0.inputs.title.set("Line Chart Example", False)
PlotLineView_0.inputs.background.set("white", False)
PlotLineView_0.inputs.plotitems.set(PlotLineItem_0.outputs.item, False)
PlotLineView_0.inputs.table.set(TableQuery_0.outputs.table, False)
PlotBarView_0.inputs.title.set("Bar Chart Example", False)
PlotBarView_0.inputs.background.set("white", False)
PlotBarView_0.inputs.plotitems.set(PlotBarItem_0.outputs.item, False)
PlotBarView_0.inputs.table.set(TableQuery_0.outputs.table, False)
CinemaDatabaseReader_0.inputs.path.set("data/plot-line.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input", False)
PlotLineItem_0.inputs.x.set("a", False)
PlotLineItem_0.inputs.y.set("b", False)
PlotLineItem_0.inputs.line.set("default", False)
PlotLineItem_0.inputs.color.set("blue", False)
PlotLineItem_0.inputs.width.set(1.0, False)
PlotBarItem_0.inputs.y.set("a", False)
PlotBarItem_0.inputs.color.set("red", False)
PlotBarItem_0.inputs.width.set(0.5, False)

# execute pipeline
TableView_0.update()
