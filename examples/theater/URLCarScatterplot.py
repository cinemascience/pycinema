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
TableView_0 = vf1.insertView( 0, pycinema.theater.views.TableView() )
ImageView_0 = vf1.insertView( 1, pycinema.theater.views.ImageView() )
vf1.setSizes([514, 531])
vf0.setSizes([1230])

# filters
CSVReader_0 = pycinema.filters.CSVReader()
Python_0 = pycinema.filters.Python()

# properties
TableView_0.inputs.table.set(CSVReader_0.outputs.table, False)
ImageView_0.inputs.images.set(Python_0.outputs.outputs, False)
CSVReader_0.inputs.path.set("https://data.wa.gov/api/views/f6w7-q2d2/rows.csv", False)
Python_0.inputs.inputs.set(CSVReader_0.outputs.table, False)
Python_0.inputs.code.set("examples/pythonfilter/URLCarScatterplot.py", False)

# execute pipeline
TableView_0.update()
