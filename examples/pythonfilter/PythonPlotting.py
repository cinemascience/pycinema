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

  outputs.append( imageFromMatplotlibFigure(figure,200) )

  plt.close(figure)


