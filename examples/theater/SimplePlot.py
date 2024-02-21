import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableView_0 = pycinema.filters.TableView()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
Plot_0 = pycinema.filters.Plot()
ImageView_0 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/plot-line.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableView_0.inputs.selection.set([], False)
PlotLineItem_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
PlotLineItem_0.inputs.x.set("a", False)
PlotLineItem_0.inputs.y.set("b", False)
PlotLineItem_0.inputs.fmt.set("x-r", False)
PlotLineItem_0.inputs.style.set({}, False)
Plot_0.inputs.items.set(PlotLineItem_0.outputs.item, False)
Plot_0.inputs.dpi.set(100, False)
ImageView_0.inputs.images.set(Plot_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame0.setSizes([1018])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame1.insertView( 0, view2 )
view4 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame1.insertView( 1, view4 )
splitFrame1.setSizes([506, 505])
tabFrame0.insertTab(1, splitFrame1)
tabFrame0.setTabText(1, 'Layout 2')
tabFrame0.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
