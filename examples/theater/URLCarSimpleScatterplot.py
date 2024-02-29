import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
CSVReader_0 = pycinema.filters.CSVReader()
TableView_0 = pycinema.filters.TableView()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
TableQuery_0 = pycinema.filters.TableQuery()
Plot_0 = pycinema.filters.Plot()
ImageView_1 = pycinema.filters.ImageView()

# properties
CSVReader_0.inputs.path.set("https://data.wa.gov/api/views/f6w7-q2d2/rows.csv", False)
TableView_0.inputs.table.set(TableQuery_0.outputs.table, False)
TableView_0.inputs.selection.set([], False)
PlotLineItem_0.inputs.table.set(TableQuery_0.outputs.table, False)
PlotLineItem_0.inputs.x.set("Model Year", False)
PlotLineItem_0.inputs.y.set("Electric Range", False)
PlotLineItem_0.inputs.fmt.set("xr", False)
PlotLineItem_0.inputs.style.set({}, False)
TableQuery_0.inputs.table.set(CSVReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input LIMIT 100", False)
Plot_0.inputs.items.set(PlotLineItem_0.outputs.item, False)
Plot_0.inputs.dpi.set(100, False)
ImageView_1.inputs.images.set(Plot_0.outputs.images, False)
ImageView_1.inputs.selection.set([], False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame1.setSizes([1431])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
view3 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame2.insertView( 0, view3 )
view5 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame2.insertView( 1, view5 )
splitFrame2.setSizes([712, 712])
tabFrame1.insertTab(1, splitFrame2)
tabFrame1.setTabText(1, 'Layout 2')
tabFrame1.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
CSVReader_0.update()
