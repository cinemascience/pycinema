import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.0.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()
TableView_0 = pycinema.filters.TableView()
ImageView_1 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set(TableView_0.inputs.selection, False)
TableView_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableView_0.inputs.selection.set([1, 2, 3, 4, 5, 6, 7, 8, 9], False)
ImageView_1.inputs.images.set(ImageView_0.outputs.images, False)
ImageView_1.inputs.selection.set([], False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame1.setSizes([1723])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
splitFrame4 = pycinema.theater.SplitFrame()
splitFrame4.setVerticalOrientation()
view2 = pycinema.theater.views.FilterView( TableView_0 )
splitFrame4.insertView( 0, view2 )
view7 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame4.insertView( 1, view7 )
splitFrame4.setSizes([513, 310])
splitFrame2.insertView( 0, splitFrame4 )
splitFrame3 = pycinema.theater.SplitFrame()
splitFrame3.setVerticalOrientation()
view5 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame3.insertView( 0, view5 )
splitFrame3.setSizes([830])
splitFrame2.insertView( 1, splitFrame3 )
splitFrame2.setSizes([706, 1010])
tabFrame1.insertTab(1, splitFrame2)
tabFrame1.setTabText(1, 'Layout 2')
tabFrame1.setCurrentIndex(1)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
CinemaDatabaseReader_0.update()
