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

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame1.setSizes([1020])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
tabFrame1.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
ImageReader_0.update()
