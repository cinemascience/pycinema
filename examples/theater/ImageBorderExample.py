import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
ImageBorder_0 = pycinema.filters.ImageBorder()
ImageView_0 = pycinema.filters.ImageView()
ImageView_1 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input WHERE phi<0", False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageBorder_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageBorder_0.inputs.width.set(10, False)
ImageBorder_0.inputs.color.set("black", False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageView_1.inputs.images.set(ImageBorder_0.outputs.images, False)
ImageView_1.inputs.selection.set([], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame0.setSizes([1299])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
view6 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 0, view6 )
view8 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame2.insertView( 1, view8 )
splitFrame2.setSizes([620, 672])
tabFrame0.insertTab(1, splitFrame2)
tabFrame0.setTabText(1, 'Layout 3')
tabFrame0.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
