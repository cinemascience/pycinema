import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
CSVReader_0 = pycinema.filters.CSVReader()
TableView_0 = pycinema.filters.TableView()
Python_0 = pycinema.filters.Python()
ImageView_0 = pycinema.filters.ImageView()

# properties
CSVReader_0.inputs.path.set("https://data.wa.gov/api/views/f6w7-q2d2/rows.csv", False)
TableView_0.inputs.table.set(CSVReader_0.outputs.table, False)
TableView_0.inputs.selection.set([], False)
Python_0.inputs.inputs.set(CSVReader_0.outputs.table, False)
Python_0.inputs.code.set("examples/pythonfilter/URLCarScatterplot.py", False)
ImageView_0.inputs.images.set(Python_0.outputs.outputs, False)
ImageView_0.inputs.selection.set([], False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame1.setSizes([2560])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setVerticalOrientation()
view2 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 0, view2 )
view5 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame2.insertView( 1, view5 )
splitFrame2.setSizes([630, 472])
tabFrame1.insertTab(1, splitFrame2)
tabFrame1.setTabText(1, 'Layout 2')
tabFrame1.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
CSVReader_0.update()
