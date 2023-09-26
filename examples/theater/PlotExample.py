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
vf3 = vf0.insertFrame(1)
vf3.setVerticalOrientation()
PlotView_2 = vf3.insertView( 0, pycinema.theater.views.PlotView() )
PlotView_3 = vf3.insertView( 1, pycinema.theater.views.PlotView() )
vf3.setSizes([421, 421])
vf2 = vf0.insertFrame(2)
vf2.setVerticalOrientation()
PlotView_4 = vf2.insertView( 0, pycinema.theater.views.PlotView() )
PlotView_6 = vf2.insertView( 1, pycinema.theater.views.PlotView() )
vf2.setSizes([421, 421])
vf0.setSizes([592, 592, 592])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/plot.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
PlotView_2.inputs.title.set("Plot Title", False)
PlotView_2.inputs.x_values.set("a", False)
PlotView_2.inputs.y_values.set("b", False)
PlotView_2.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotView_3.inputs.title.set("Plot Title", False)
PlotView_3.inputs.x_values.set("a", False)
PlotView_3.inputs.y_values.set("c", False)
PlotView_3.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotView_4.inputs.title.set("Plot Title", False)
PlotView_4.inputs.x_values.set("b", False)
PlotView_4.inputs.y_values.set("a", False)
PlotView_4.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotView_6.inputs.title.set("Plot Title", False)
PlotView_6.inputs.x_values.set("b", False)
PlotView_6.inputs.y_values.set("c", False)
PlotView_6.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)

# execute pipeline
CinemaDatabaseReader_0.update()
