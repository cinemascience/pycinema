import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

# filters
ImageReader_0 = pycinema.filters.ImageReader()
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
InterpolationCompression_0 = pycinema.filters.InterpolationCompression()
ImageBorder_0 = pycinema.filters.ImageBorder()
ImageAnnotation_0 = pycinema.filters.ImageAnnotation()
ImageView_0 = pycinema.filters.ImageView()
ImageBorder_1 = pycinema.filters.ImageBorder()
VideoWriter_0 = pycinema.filters.VideoWriter()

# properties
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
CinemaDatabaseReader_0.inputs.path.set("/home/jones/projects/cinema-lib/pycinema-error/data/time_sequences/volume_stride1.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input LIMIT 5", False)
InterpolationCompression_0.inputs.images.set(ImageReader_0.outputs.images, False)
InterpolationCompression_0.inputs.model_path.set("/home/jones/projects/cinema-lib/pycinema-inter/data/film_net_fp32.pt", False)
InterpolationCompression_0.inputs.error.set(0.99, False)
ImageBorder_0.inputs.images.set(InterpolationCompression_0.outputs.images, False)
ImageBorder_0.inputs.width.set(20, False)
ImageBorder_0.inputs.color.set("red", False)
ImageBorder_0.inputs.condition.set("interpolated", False)
ImageAnnotation_0.inputs.images.set(ImageBorder_1.outputs.images, False)
ImageAnnotation_0.inputs.xy.set((20, 20), False)
ImageAnnotation_0.inputs.size.set(20, False)
ImageAnnotation_0.inputs.spacing.set(0, False)
ImageAnnotation_0.inputs.color.set((), False)
ImageAnnotation_0.inputs.ignore.set(['^file', '^id'], False)
ImageView_0.inputs.images.set(ImageAnnotation_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageBorder_1.inputs.images.set(ImageBorder_0.outputs.images, False)
ImageBorder_1.inputs.width.set(20, False)
ImageBorder_1.inputs.color.set("green", False)
ImageBorder_1.inputs.condition.set("not interpolated", False)
VideoWriter_0.inputs.images.set(ImageAnnotation_0.outputs.images, False)
VideoWriter_0.inputs.path.set("/tmp/test.mp4", False)
VideoWriter_0.inputs.fps.set(3, False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
view3 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame1.insertView( 1, view3 )
splitFrame1.setSizes([508, 508])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
tabFrame1.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
ImageReader_0.update()
