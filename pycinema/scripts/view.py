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
ParameterView_0 = vf1.insertView( 0, pycinema.theater.views.ParameterView() )
TableView_0 = vf1.insertView( 1, pycinema.theater.views.TableView() )
ColorMappingView_0 = vf1.insertView( 2, pycinema.theater.views.ColorMappingView() )
ImageView_0 = vf0.insertView( 1, pycinema.theater.views.ImageView() )
vf0.setSizes([300,600])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()
ShaderSSAO_0 = pycinema.filters.ShaderSSAO()
ImageAnnotation_0 = pycinema.filters.ImageAnnotation()

# properties
ParameterView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)

# properties
CinemaDatabaseReader_0.inputs.path.set(PYCINEMA_ARG_0, False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableView_0.inputs.table.set(ParameterView_0.outputs.table, False)
ImageReader_0.inputs.table.set(ParameterView_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.compose.set(ParameterView_0.outputs.compose, False)
ColorMappingView_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ShaderSSAO_0.inputs.images.set(ColorMappingView_0.outputs.images,False)
ShaderSSAO_0.inputs.samples.set(128,False)
ImageAnnotation_0.inputs.images.set(ShaderSSAO_0.outputs.images, False)
ImageView_0.inputs.images.set(ImageAnnotation_0.outputs.images, False)

# execute pipeline
CinemaDatabaseReader_0.update()
