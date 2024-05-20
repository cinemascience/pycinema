import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageConvertGrayscale_0 = pycinema.filters.ImageConvertGrayscale()
ImageView_0 = pycinema.filters.ImageView()
ImageView_1 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input WHERE phi<0", False)
ImageConvertGrayscale_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.images.set(ImageConvertGrayscale_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageView_1.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_1.inputs.selection.set([], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame0.setSizes([1701])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame1.insertView( 0, view2 )
view4 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame1.insertView( 1, view4 )
splitFrame1.setSizes([848, 846])
tabFrame0.insertTab(1, splitFrame1)
tabFrame0.setTabText(1, 'Layout 2')
tabFrame0.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
