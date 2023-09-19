import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '1.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf1 = vf0.insertFrame(1)
vf1.setVerticalOrientation()
ImageView_0 = vf1.insertView( 0, pycinema.theater.views.ImageView() )
ImageView_1 = vf1.insertView( 1, pycinema.theater.views.ImageView() )
vf1.setSizes([421, 421])
vf0.setSizes([1135, 490])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
ImageCanny_0 = pycinema.filters.ImageCanny()
MaskCompositing_0 = pycinema.filters.MaskCompositing()
ColorSource_0 = pycinema.filters.ColorSource()
CinemaDatabaseWriter_0 = pycinema.filters.CinemaDatabaseWriter()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input where phi=-180", False)
ImageView_0.inputs.images.set(ImageCanny_0.outputs.images, False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageCanny_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageCanny_0.inputs.thresholds.set((100, 150), False)
MaskCompositing_0.inputs.images_a.set(ImageCanny_0.outputs.images, False)
MaskCompositing_0.inputs.images_b.set(ColorSource_0.outputs.rgba, False)
MaskCompositing_0.inputs.masks.set(ImageCanny_0.outputs.images, False)
MaskCompositing_0.inputs.color_channel.set("rgba", False)
MaskCompositing_0.inputs.mask_channel.set("canny", False)
MaskCompositing_0.inputs.opacity.set(0.5, False)
ColorSource_0.inputs.rgba.set((255, 255, 255, 255), False)
ImageView_1.inputs.images.set(MaskCompositing_0.outputs.images, False)
CinemaDatabaseWriter_0.inputs.images.set(MaskCompositing_0.outputs.images, False)
CinemaDatabaseWriter_0.inputs.path.set("cann.cdb", False)
CinemaDatabaseWriter_0.inputs.ignore.set(['^id', '^camera', '^FILE'], False)

# execute pipeline
CinemaDatabaseReader_0.update()
