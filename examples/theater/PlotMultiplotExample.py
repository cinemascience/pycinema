import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf2 = vf0.insertFrame(0)
vf2.setVerticalOrientation()
vf2.insertView( 0, pycinema.theater.views.NodeEditorView() )
TableView_0 = vf2.insertView( 1, pycinema.theater.views.TableView() )
vf2.setSizes([603, 378])
vf1 = vf0.insertFrame(1)
vf1.setVerticalOrientation()
PlotLineView_0 = vf1.insertView( 0, pycinema.theater.views.PlotLineView() )
vf1.setSizes([988])
vf0.setSizes([777, 906])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
PlotLineItem_1 = pycinema.filters.PlotLineItem()
CinemaDatabaseReader_1 = pycinema.filters.CinemaDatabaseReader()

# properties
PlotLineView_0.inputs.title.set("Multiplot Test ", False)
PlotLineView_0.inputs.background.set("white", False)
PlotLineView_0.inputs.plotitem.set([PlotLineItem_0.outputs.plotitem,PlotLineItem_1.outputs.plotitem], False)
CinemaDatabaseReader_0.inputs.path.set("data/plot-line.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
PlotLineItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotLineItem_0.inputs.x.set("a", False)
PlotLineItem_0.inputs.y.set("b", False)
PlotLineItem_0.inputs.penstyle.set("dash", False)
PlotLineItem_0.inputs.pencolor.set("red", False)
PlotLineItem_0.inputs.penwidth.set(2.0, False)
PlotLineItem_1.inputs.table.set(CinemaDatabaseReader_1.outputs.table, False)
PlotLineItem_1.inputs.x.set("c", False)
PlotLineItem_1.inputs.y.set("b", False)
PlotLineItem_1.inputs.penstyle.set("default", False)
PlotLineItem_1.inputs.pencolor.set("blue", False)
PlotLineItem_1.inputs.penwidth.set(1.0, False)
TableView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
CinemaDatabaseReader_1.inputs.path.set("data/plot-line.cdb", False)
CinemaDatabaseReader_1.inputs.file_column.set("FILE", False)

# execute pipeline
PlotLineView_0.update()
