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
vf2 = vf1.insertFrame(0)
vf2.setHorizontalOrientation()
vf3 = vf2.insertFrame(0)
vf3.setVerticalOrientation()
vf4 = vf3.insertFrame(0)
vf4.setHorizontalOrientation()
TableView_0 = vf4.insertView( 0, pycinema.theater.views.TableView() )
ImageView_0 = vf4.insertView( 1, pycinema.theater.views.ImageView() )
vf4.setSizes([771, 771])
vf5 = vf3.insertFrame(1)
vf5.setHorizontalOrientation()
vf5.insertView( 0, pycinema.theater.views.NodeEditorView() )
PlotScatterView_0 = vf5.insertView( 1, pycinema.theater.views.PlotScatterView() )
vf5.setSizes([771, 771])
vf3.setSizes([513, 447])
vf2.setSizes([1549])
vf1.setSizes([967])
vf0.setSizes([1549])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
MLTFReader_0 = pycinema.filters.MLTFReader()
MLTFPredictor_0 = pycinema.filters.MLTFPredictor()
ImagesToTable_0 = pycinema.filters.ImagesToTable()
PlotScatterItem_0 = pycinema.filters.PlotScatterItem()

# properties
TableView_0.inputs.table.set(MLTFPredictor_0.outputs.images, False)
ImageView_0.inputs.images.set(MLTFPredictor_0.outputs.images, False)
PlotScatterView_0.inputs.title.set("MNIST Comparative Plot", False)
PlotScatterView_0.inputs.background.set("white", False)
PlotScatterView_0.inputs.plotitem.set(PlotScatterItem_0.outputs.plotitem, False)
CinemaDatabaseReader_0.inputs.path.set("data/mnist.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
MLTFReader_0.inputs.path.set("data/MNIST_models/TF/mnist_tf.h5", False)
MLTFPredictor_0.inputs.trainedModel.set(MLTFReader_0.outputs.model, False)
MLTFPredictor_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImagesToTable_0.inputs.images.set(MLTFPredictor_0.outputs.images, False)
PlotScatterItem_0.inputs.table.set(ImagesToTable_0.outputs.table, False)
PlotScatterItem_0.inputs.x.set("TrueValue", False)
PlotScatterItem_0.inputs.y.set("PredictedValue", False)
PlotScatterItem_0.inputs.pencolor.set("default", False)
PlotScatterItem_0.inputs.penwidth.set(2.0, False)
PlotScatterItem_0.inputs.brushcolor.set("yellow", False)
PlotScatterItem_0.inputs.symbol.set("o", False)
PlotScatterItem_0.inputs.size.set(10.0, False)

# execute pipeline
TableView_0.update()
