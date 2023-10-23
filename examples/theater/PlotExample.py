import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.0.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf1 = vf0.insertFrame(1)
vf1.setVerticalOrientation()
PlotLineView_0 = vf1.insertView( 0, pycinema.theater.views.PlotLineView() )
PlotBarView_0 = vf1.insertView( 1, pycinema.theater.views.PlotBarView() )
vf1.setSizes([425, 424])
vf0.setSizes([848, 847])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
PlotBarItem_0 = pycinema.filters.PlotBarItem()

# properties
PlotLineView_0.inputs.title.set("Sample Line Chart", False)
PlotLineView_0.inputs.background.set("white", False)
PlotLineView_0.inputs.plotitem.set(PlotLineItem_0.outputs.plotitem, False)
PlotBarView_0.inputs.title.set("Plot Title", False)
PlotBarView_0.inputs.background.set("white", False)
PlotBarView_0.inputs.plotitem.set(PlotBarItem_0.outputs.plotitem, False)
CinemaDatabaseReader_0.inputs.path.set("data/plot-line.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
PlotLineItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotLineItem_0.inputs.x.set("a", False)
PlotLineItem_0.inputs.y.set("b", False)
PlotLineItem_0.inputs.linetype.set("default", False)
PlotLineItem_0.inputs.linecolor.set("red", False)
PlotLineItem_0.inputs.linewidth.set(1.0, False)
PlotBarItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotBarItem_0.inputs.y.set("a", False)
PlotBarItem_0.inputs.barcolor.set("blue", False)
PlotBarItem_0.inputs.barwidth.set(0.5, False)

# execute pipeline
PlotLineView_0.update()
