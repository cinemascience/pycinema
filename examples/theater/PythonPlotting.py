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
vf1.setHorizontalOrientation()
vf2 = vf1.insertFrame(0)
vf2.setVerticalOrientation()
vf2.insertView( 0, pycinema.theater.views.NodeEditorView() )
TableView_0 = vf2.insertView( 1, pycinema.theater.views.TableView() )
vf2.setSizes([425, 425])
vf3 = vf1.insertFrame(1)
vf3.setVerticalOrientation()
ImageView_0 = vf3.insertView( 0, pycinema.theater.views.ImageView() )
TextView_0 = vf3.insertView( 1, pycinema.theater.views.TextView() )
vf3.setSizes([404, 446])
vf1.setSizes([510, 510])
vf0.setSizes([1024])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
Calculator_0 = pycinema.filters.Calculator()
TableQuery_0 = pycinema.filters.TableQuery()
Python_0 = pycinema.filters.Python()

# properties
TableView_0.inputs.table.set(Calculator_0.outputs.table, False)
ImageView_0.inputs.images.set(Python_0.outputs.outputs, False)
CinemaDatabaseReader_0.inputs.path.set("./data/ScalarImages.cdb/", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
Calculator_0.inputs.table.set(TableQuery_0.outputs.table, False)
Calculator_0.inputs.label.set("result", False)
Calculator_0.inputs.expression.set("time*id", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input", False)
Python_0.inputs.inputs.set(Calculator_0.outputs.table, False)
Python_0.inputs.code.set(TextView_0.outputs.text, False)
TextView_0.inputs.text.set('''
from pycinema import getColumnFromTable, imageFromMatplotlibFigure
import matplotlib.pyplot as plt

# use offscreen backend
import matplotlib as mpl
mpl.use('Agg')

outputs = []
for column in ['result','time','phi','theta']:

  figure = plt.figure()
  plt.plot(
    getColumnFromTable(inputs, "id"),
    getColumnFromTable(inputs, column)
  )
  plt.xlabel('id')
  plt.ylabel(column)
  plt.title('id x ' + column)

  outputs.append( imageFromMatplotlibFigure(figure) )

  plt.close(figure)

''', False)

# execute pipeline
TableView_0.update()
