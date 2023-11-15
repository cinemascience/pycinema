import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf1 = vf0.insertFrame(0)
vf1.setVerticalOrientation()
vf1.insertView( 0, pycinema.theater.views.NodeEditorView() )
TableView_0 = vf1.insertView( 1, pycinema.theater.views.TableView() )
vf1.setSizes([400, 442])
vf2 = vf0.insertFrame(1)
vf2.setVerticalOrientation()
PlotLineView_0 = vf2.insertView( 0, pycinema.theater.views.PlotLineView() )
PlotLineView_1 = vf2.insertView( 1, pycinema.theater.views.PlotLineView() )
vf2.setSizes([400, 442])
vf0.setSizes([637, 636])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
PlotLineItem_1 = pycinema.filters.PlotLineItem()

# properties
TableView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotLineView_0.inputs.title.set("Plot Title", False)
PlotLineView_0.inputs.background.set("white", False)
PlotLineView_0.inputs.plotitem.set(PlotLineItem_0.outputs.plotitem, False)
CinemaDatabaseReader_0.inputs.path.set("data/plot-line.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
PlotLineItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotLineItem_0.inputs.x.set("a", False)
PlotLineItem_0.inputs.y.set("b", False)
PlotLineItem_0.inputs.penstyle.set("default", False)
PlotLineItem_0.inputs.pencolor.set("default", False)
PlotLineItem_0.inputs.penwidth.set(1.0, False)
PlotLineItem_1.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotLineItem_1.inputs.x.set("a", False)
PlotLineItem_1.inputs.y.set("d", False)
PlotLineItem_1.inputs.penstyle.set("default", False)
PlotLineItem_1.inputs.pencolor.set("default", False)
PlotLineItem_1.inputs.penwidth.set(1.0, False)
PlotLineView_1.inputs.title.set("Plot Title", False)
PlotLineView_1.inputs.background.set("white", False)
PlotLineView_1.inputs.plotitem.set(PlotLineItem_1.outputs.plotitem, False)

# execute pipeline
TableView_0.update()
