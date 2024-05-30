import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# this application settings
EXPLORE = { 'VERSION' : '1.0'}

# reporting
print("explorer v" + EXPLORE["VERSION"])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ParallelCoordinates_0 = pycinema.filters.ParallelCoordinates()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()
ColorMapping_0 = pycinema.filters.ColorMapping()
ShaderSSAO_0 = pycinema.filters.ShaderSSAO()
ImageAnnotation_0 = pycinema.filters.ImageAnnotation()
TableView_0 = pycinema.filters.TableView()
ImageView_0 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set(PYCINEMA_ARG_0, False)
ParallelCoordinates_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.table.set(ParallelCoordinates_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.compose.set(ParallelCoordinates_0.outputs.compose, False)
ColorMapping_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ShaderSSAO_0.inputs.images.set(ColorMapping_0.outputs.images, False)
ImageAnnotation_0.inputs.images.set(ShaderSSAO_0.outputs.images, False)
TableView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageView_0.inputs.images.set(ImageAnnotation_0.outputs.images, False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setVerticalOrientation()
view5 = pycinema.theater.views.FilterView( ParallelCoordinates_0 )
splitFrame1.insertView( 0, view5 )
view6 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame1.insertView( 1, view6 )
view7 = pycinema.theater.views.FilterView( ColorMapping_0 )
splitFrame1.insertView( 2, view7 )
splitFrame1.setSizes([333, 333, 333])
splitFrame0.insertView( 0, splitFrame1 )
view8 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame0.insertView( 1, view8 )
splitFrame0.setSizes([500, 500])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
tabFrame0.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
