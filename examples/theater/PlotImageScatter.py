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
vf1.insertView( 0, pycinema.theater.views.NodeEditorView() )
InspectorView_0 = vf1.insertView( 1, pycinema.theater.views.InspectorView() )
vf1.setSizes([400, 442])
vf2 = vf0.insertFrame(1)
vf2.setVerticalOrientation()
PlotScatterView_0 = vf2.insertView( 0, pycinema.theater.views.PlotScatterView() )
ImageView_0 = vf2.insertView( 1, pycinema.theater.views.ImageView() )
vf2.setSizes([400, 442])
vf0.setSizes([671, 671])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageMetadataToScatterItem_0 = pycinema.filters.ImageMetadataToScatterItem()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
InspectorView_0.inputs.object.set(ImageReader_0.outputs.images, False)
ImageMetadataToScatterItem_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageMetadataToScatterItem_0.inputs.x.set("phi", False)
ImageMetadataToScatterItem_0.inputs.y.set("theta", False)
ImageMetadataToScatterItem_0.inputs.pencolor.set("default", False)
ImageMetadataToScatterItem_0.inputs.penwidth.set(1.0, False)
ImageMetadataToScatterItem_0.inputs.brushcolor.set("default", False)
ImageMetadataToScatterItem_0.inputs.symbol.set("x", False)
ImageMetadataToScatterItem_0.inputs.size.set(1.0, False)
PlotScatterView_0.inputs.title.set("Plot Title", False)
PlotScatterView_0.inputs.background.set("white", False)
PlotScatterView_0.inputs.plotitem.set(ImageMetadataToScatterItem_0.outputs.plotitem, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)

# execute pipeline
CinemaDatabaseReader_0.update()
