import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.2.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()
ImageBorder_0 = pycinema.filters.ImageBorder()
TableQuery_0 = pycinema.filters.TableQuery()

# properties
CinemaDatabaseReader_0.inputs.path.set("/Users/dhr/LANL/dev/cinemascience/pycinema/data/playingcards.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageBorder_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageBorder_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageBorder_0.inputs.width.set(1, False)
ImageBorder_0.inputs.color.set("AUTO", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input WHERE value > 9", False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame0.insertView( 0, view2 )
splitFrame0.setSizes([919])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
tabFrame0.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
