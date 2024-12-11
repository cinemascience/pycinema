import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

# filters
TableReader_0 = pycinema.filters.TableReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()

# properties
TableReader_0.inputs.path.set(PYCINEMA_ARG_0, False)
TableReader_0.inputs.file_column.set("FILE", False)
TableReader_0.update()
TableQuery_0.inputs.table.set(TableReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input LIMIT 100", False)

# image reader
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set(TableReader_0.inputs.file_column, False)
ImageReader_0.inputs.cache.set(True, False)

# optional recoloring
ColorMapping_0 = None
if TableReader_0.istype("hdf5"):
    ColorMapping_0 = pycinema.filters.ColorMapping()
    ColorMapping_0.inputs.map.set("grey", False)
    ColorMapping_0.inputs.nan.set((1, 1, 1, 1), False)
    ColorMapping_0.inputs.range.set((0, 1), False)
    # TODO: get the first available channel
    ColorMapping_0.inputs.channel.set("Depth", False)
    ColorMapping_0.inputs.images.set(ImageReader_0.outputs.images, False)
    ColorMapping_0.inputs.composition_id.set(-1, False)
    ImageView_0.inputs.images.set(ColorMapping_0.outputs.images, False)
else:
    ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view9 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame1.insertView( 0, view9 )
splitFrame1.setSizes([1018])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
tabFrame1.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
TableReader_0.update()
