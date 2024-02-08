import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setVerticalOrientation()
vf3 = vf0.insertFrame(0)
vf3.setHorizontalOrientation()
vf3.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf4 = vf3.insertFrame(1)
vf4.setVerticalOrientation()
PlotLineView_2 = vf4.insertView( 0, pycinema.theater.views.PlotLineView() )
PlotScatterView_2 = vf4.insertView( 1, pycinema.theater.views.PlotScatterView() )
vf4.setSizes([455, 454])
vf3.setSizes([718, 717])
vf0.setSizes([916])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
PlotScatterItem_0 = pycinema.filters.PlotScatterItem()
PlotLineItem_1 = pycinema.filters.PlotLineItem()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/plot-line.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
PlotLineItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotLineItem_0.inputs.x.set("a", False)
PlotLineItem_0.inputs.y.set("b", False)
PlotLineItem_0.inputs.penstyle.set("default", False)
PlotLineItem_0.inputs.pencolor.set("default", False)
PlotLineItem_0.inputs.penwidth.set(1.0, False)
PlotScatterItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotScatterItem_0.inputs.x.set("b", False)
PlotScatterItem_0.inputs.y.set("c", False)
PlotScatterItem_0.inputs.pencolor.set("red", False)
PlotScatterItem_0.inputs.penwidth.set(1.0, False)
PlotScatterItem_0.inputs.brushcolor.set("default", False)
PlotScatterItem_0.inputs.symbol.set("o", False)
PlotScatterItem_0.inputs.size.set(10.0, False)
PlotLineItem_1.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotLineItem_1.inputs.x.set("b", False)
PlotLineItem_1.inputs.y.set("a", False)
PlotLineItem_1.inputs.penstyle.set("default", False)
PlotLineItem_1.inputs.pencolor.set("default", False)
PlotLineItem_1.inputs.penwidth.set(1.0, False)
PlotLineView_2.inputs.title.set("Plot Title", False)
PlotLineView_2.inputs.background.set("white", False)
PlotLineView_2.inputs.plotitem.set([PlotLineItem_1.outputs.plotitem,PlotLineItem_0.outputs.plotitem], False)
PlotScatterView_2.inputs.title.set("Plot Title", False)
PlotScatterView_2.inputs.background.set("white", False)
PlotScatterView_2.inputs.plotitem.set(PlotScatterItem_0.outputs.plotitem, False)

# execute pipeline
CinemaDatabaseReader_0.update()
