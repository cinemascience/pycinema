import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

# filters
TableReader_0 = pycinema.filters.TableReader()
InspectorView_0 = pycinema.filters.InspectorView()

# properties
TableReader_0.inputs.path.set("data/wildfire/wildfire.db", False)
TableReader_0.inputs.file_column.set("FILE", False)
InspectorView_0.inputs.object.set(TableReader_0.outputs.table, False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setVerticalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
view4 = pycinema.theater.views.FilterView( InspectorView_0 )
splitFrame0.insertView( 1, view4 )
splitFrame0.setSizes([305, 508])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
tabFrame0.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
TableReader_0.update()
