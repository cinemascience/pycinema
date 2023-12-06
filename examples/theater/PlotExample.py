import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf1 = vf0.insertFrame(1)
vf1.setVerticalOrientation()
PlotBarView_0 = vf1.insertView( 0, pycinema.theater.views.PlotBarView() )
PlotLineView_0 = vf1.insertView( 1, pycinema.theater.views.PlotLineView() )
PlotScatterView_0 = vf1.insertView( 2, pycinema.theater.views.PlotScatterView() )
vf1.setSizes([292, 288, 295])
vf0.setSizes([706, 705])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
PlotBarItem_0 = pycinema.filters.PlotBarItem()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
PlotScatterItem_0 = pycinema.filters.PlotScatterItem()

# properties
PlotBarView_0.inputs.title.set("Plot Title", False)
PlotBarView_0.inputs.background.set("white", False)
PlotBarView_0.inputs.plotitem.set(PlotBarItem_0.outputs.plotitem, False)
PlotLineView_0.inputs.title.set("Plot Title", False)
PlotLineView_0.inputs.background.set("white", False)
PlotLineView_0.inputs.plotitem.set(PlotLineItem_0.outputs.plotitem, False)
PlotScatterView_0.inputs.title.set("Plot Title", False)
PlotScatterView_0.inputs.background.set("white", False)
PlotScatterView_0.inputs.plotitem.set(PlotScatterItem_0.outputs.plotitem, False)
CinemaDatabaseReader_0.inputs.path.set("data/plot-line.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
PlotBarItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotBarItem_0.inputs.x.set("a", False)
PlotBarItem_0.inputs.y.set("b", False)
PlotBarItem_0.inputs.brushcolor.set("blue", False)
PlotBarItem_0.inputs.width.set(0.5, False)
PlotLineItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotLineItem_0.inputs.x.set("a", False)
PlotLineItem_0.inputs.y.set("b", False)
PlotLineItem_0.inputs.penstyle.set("dash", False)
PlotLineItem_0.inputs.pencolor.set("red", False)
PlotLineItem_0.inputs.penwidth.set(2.0, False)
PlotScatterItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotScatterItem_0.inputs.x.set("a", False)
PlotScatterItem_0.inputs.y.set("b", False)
PlotScatterItem_0.inputs.pencolor.set("black", False)
PlotScatterItem_0.inputs.penwidth.set(1.0, False)
PlotScatterItem_0.inputs.brushcolor.set("gray", False)
PlotScatterItem_0.inputs.symbol.set("o", False)
PlotScatterItem_0.inputs.size.set(5.0, False)

# execute pipeline
PlotBarView_0.update()
