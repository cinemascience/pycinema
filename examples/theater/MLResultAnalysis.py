import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.0.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
MLTFReader_0 = pycinema.filters.MLTFReader()
MLTFPredictor_0 = pycinema.filters.MLTFPredictor()
ImageView_0 = pycinema.filters.ImageView()
ImagesToTable_0 = pycinema.filters.ImagesToTable()
TableView_0 = pycinema.filters.TableView()
PlotLineItem_0 = pycinema.filters.PlotLineItem()
Plot_0 = pycinema.filters.Plot()
ImageView_2 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/mnist.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
MLTFReader_0.inputs.path.set("data/MNIST_models/TF/mnist_tf.h5", False)
MLTFPredictor_0.inputs.trainedModel.set(MLTFReader_0.outputs.model, False)
MLTFPredictor_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImagesToTable_0.inputs.images.set(MLTFPredictor_0.outputs.images, False)
TableView_0.inputs.table.set(ImagesToTable_0.outputs.table, False)
TableView_0.inputs.selection.set([], False)
PlotLineItem_0.inputs.table.set(ImagesToTable_0.outputs.table, False)
PlotLineItem_0.inputs.x.set("TrueValue", False)
PlotLineItem_0.inputs.y.set("PredictedValue_0", False)
PlotLineItem_0.inputs.fmt.set("ro", False)
PlotLineItem_0.inputs.style.set({}, False)
Plot_0.inputs.items.set(PlotLineItem_0.outputs.item, False)
Plot_0.inputs.dpi.set(100, False)
ImageView_2.inputs.images.set(Plot_0.outputs.images, False)
ImageView_2.inputs.selection.set([], False)

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
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setVerticalOrientation()
view4 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 0, view4 )
view7 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame2.insertView( 1, view7 )
splitFrame2.setSizes([292, 508])
splitFrame1.insertView( 0, splitFrame2 )
view8 = pycinema.theater.views.FilterView( ImageView_2 )
splitFrame1.insertView( 1, view8 )
splitFrame1.setSizes([506, 505])
tabFrame0.insertTab(1, splitFrame1)
tabFrame0.setTabText(1, 'Layout 2')
tabFrame0.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
