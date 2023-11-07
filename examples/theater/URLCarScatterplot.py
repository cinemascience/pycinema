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
TableView_1 = vf1.insertView( 1, pycinema.theater.views.TableView() )
vf1.setSizes([610, 440])
PlotScatterView_0 = vf0.insertView( 1, pycinema.theater.views.PlotScatterView() )
vf0.setSizes([1111, 1110])

# filters
PlotScatterItem_0 = pycinema.filters.PlotScatterItem()
CSVReader_0 = pycinema.filters.CSVReader()

# properties
PlotScatterView_0.inputs.title.set("Electric Range vs. Model Year for Vehicles in Washington State (2023)", False)
PlotScatterView_0.inputs.background.set("white", False)
PlotScatterView_0.inputs.plotitem.set(PlotScatterItem_0.outputs.plotitem, False)
PlotScatterItem_0.inputs.table.set(CSVReader_0.outputs.table, False)
PlotScatterItem_0.inputs.x.set("Model Year", False)
PlotScatterItem_0.inputs.y.set("Electric Range", False)
PlotScatterItem_0.inputs.pencolor.set("default", False)
PlotScatterItem_0.inputs.penwidth.set(1.0, False)
PlotScatterItem_0.inputs.brushcolor.set("light blue", False)
PlotScatterItem_0.inputs.symbol.set("o", False)
PlotScatterItem_0.inputs.size.set(10.0, False)
TableView_1.inputs.table.set(CSVReader_0.outputs.table, False)
CSVReader_0.inputs.path.set("https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD", False)

# execute pipeline
PlotScatterView_0.update()
