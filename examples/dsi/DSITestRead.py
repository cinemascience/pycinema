import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views
import os

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

cdbpath = "DSIscratch/sphere.cdb"

# filters
DSICinemaDatabaseReader_0 = pycinema.filters.DSICinemaDatabaseReader()
DSIReader_0 = pycinema.filters.DSIReader()
InspectorView_0 = pycinema.filters.InspectorView()
InspectorView_1 = pycinema.filters.InspectorView()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()
ImageReader_1 = pycinema.filters.ImageReader()
ImageView_1 = pycinema.filters.ImageView()

# properties
DSICinemaDatabaseReader_0.inputs.path.set(cdbpath, False)
DSICinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
DSIReader_0.inputs.path.set(os.path.join(cdbpath, "data.db"), False)
DSIReader_0.inputs.tablename.set("datacsv", False)
InspectorView_0.inputs.object.set(DSIReader_0.outputs.table, False)
InspectorView_1.inputs.object.set(DSICinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.table.set(DSIReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageReader_1.inputs.table.set(DSICinemaDatabaseReader_0.outputs.table, False)
ImageReader_1.inputs.file_column.set("FILE", False)
ImageReader_1.inputs.cache.set(True, False)
ImageView_1.inputs.images.set(ImageReader_1.outputs.images, False)
ImageView_1.inputs.selection.set([], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setVerticalOrientation()
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( InspectorView_0 )
splitFrame2.insertView( 0, view2 )
view6 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 1, view6 )
splitFrame2.setSizes([361, 360])
splitFrame1.insertView( 0, splitFrame2 )
splitFrame3 = pycinema.theater.SplitFrame()
splitFrame3.setHorizontalOrientation()
view4 = pycinema.theater.views.FilterView( InspectorView_1 )
splitFrame3.insertView( 0, view4 )
view8 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame3.insertView( 1, view8 )
splitFrame3.setSizes([361, 360])
splitFrame1.insertView( 1, splitFrame3 )
splitFrame1.setSizes([409, 412])
splitFrame0.insertView( 1, splitFrame1 )
splitFrame0.setSizes([729, 728])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
tabFrame0.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
DSICinemaDatabaseReader_0.update()
