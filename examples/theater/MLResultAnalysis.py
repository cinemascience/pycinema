import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
MLTFReader_0 = pycinema.filters.MLTFReader()
MLTFPredictor_0 = pycinema.filters.MLTFPredictor()
ImagesToTable_0 = pycinema.filters.ImagesToTable()
TableView_0 = pycinema.filters.TableView()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
Plot_0 = pycinema.filters.Plot()
ImageView_0 = pycinema.filters.ImageView()
ImageView_1 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/mnist.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
MLTFReader_0.inputs.path.set("data/MNIST_models/TF/mnist_tf.h5", False)
MLTFPredictor_0.inputs.trainedModel.set(MLTFReader_0.outputs.model, False)
MLTFPredictor_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImagesToTable_0.inputs.images.set(MLTFPredictor_0.outputs.images, False)
TableView_0.inputs.table.set(ImagesToTable_0.outputs.table, False)
TableView_0.inputs.selection.set([], False)
PlotLineItem_0.inputs.table.set(ImagesToTable_0.outputs.table, False)
PlotLineItem_0.inputs.x.set("TrueValue", False)
PlotLineItem_0.inputs.y.set("PredictedValue", False)
PlotLineItem_0.inputs.fmt.set("", False)
PlotLineItem_0.inputs.style.set({}, False)
Plot_0.inputs.items.set(PlotLineItem_0.outputs.item, False)
Plot_0.inputs.dpi.set(100, False)
ImageView_0.inputs.images.set(Plot_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageView_1.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_1.inputs.selection.set([], False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame1.setSizes([1018])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
splitFrame4 = pycinema.theater.SplitFrame()
splitFrame4.setHorizontalOrientation()
view13 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame4.insertView( 0, view13 )
view15 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame4.insertView( 1, view15 )
splitFrame4.setSizes([506, 505])
tabFrame1.insertTab(1, splitFrame4)
tabFrame1.setTabText(1, 'Layout 3')
tabFrame1.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
CinemaDatabaseReader_0.update()
